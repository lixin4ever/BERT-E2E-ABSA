# coding=utf-8
# Copyright 2018 Google AI Language, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import logging
import os
logger = logging.getLogger(__name__)


def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    """ A map of modules from TF to PyTorch.
        I use a map to keep the PyTorch model as
        identical to the original PyTorch model as possible.
    """
    tf_to_pt_map = {}

    if hasattr(model, 'transformer'):
        if hasattr(model, 'lm_loss'):
            # We will load also the output bias
            tf_to_pt_map['model/lm_loss/bias'] = model.lm_loss.bias
        if hasattr(model, 'sequence_summary') and 'model/sequnece_summary/summary/kernel' in tf_weights:
            # We will load also the sequence summary
            tf_to_pt_map['model/sequnece_summary/summary/kernel'] = model.sequence_summary.summary.weight
            tf_to_pt_map['model/sequnece_summary/summary/bias'] = model.sequence_summary.summary.bias
        if hasattr(model, 'logits_proj') and config.finetuning_task is not None \
                and 'model/regression_{}/logit/kernel'.format(config.finetuning_task) in tf_weights:
            tf_to_pt_map['model/regression_{}/logit/kernel'.format(config.finetuning_task)] = model.logits_proj.weight
            tf_to_pt_map['model/regression_{}/logit/bias'.format(config.finetuning_task)] = model.logits_proj.bias

        # Now load the rest of the transformer
        model = model.transformer

    # Embeddings and output
    tf_to_pt_map.update({'model/transformer/word_embedding/lookup_table': model.word_embedding.weight,
                             'model/transformer/mask_emb/mask_emb': model.mask_emb})

    # Transformer blocks
    for i, b in enumerate(model.layer):
        layer_str = "model/transformer/layer_%d/" % i
        tf_to_pt_map.update({
            layer_str + "rel_attn/LayerNorm/gamma": b.rel_attn.layer_norm.weight,
            layer_str + "rel_attn/LayerNorm/beta": b.rel_attn.layer_norm.bias,
            layer_str + "rel_attn/o/kernel": b.rel_attn.o,
            layer_str + "rel_attn/q/kernel": b.rel_attn.q,
            layer_str + "rel_attn/k/kernel": b.rel_attn.k,
            layer_str + "rel_attn/r/kernel": b.rel_attn.r,
            layer_str + "rel_attn/v/kernel": b.rel_attn.v,
            layer_str + "ff/LayerNorm/gamma": b.ff.layer_norm.weight,
            layer_str + "ff/LayerNorm/beta": b.ff.layer_norm.bias,
            layer_str + "ff/layer_1/kernel": b.ff.layer_1.weight,
            layer_str + "ff/layer_1/bias": b.ff.layer_1.bias,
            layer_str + "ff/layer_2/kernel": b.ff.layer_2.weight,
            layer_str + "ff/layer_2/bias": b.ff.layer_2.bias,
        })

    # Relative positioning biases
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]

    tf_to_pt_map.update({
        'model/transformer/r_r_bias': r_r_list,
        'model/transformer/r_w_bias': r_w_list,
        'model/transformer/r_s_bias': r_s_list,
        'model/transformer/seg_embed': seg_embed_list})
    return tf_to_pt_map


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def load_tf_weights_in_xlnet(model, config, tf_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. "
                     "Please see https://www.tensorflow.org/install/ for installation instructions.")

    # load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        tf_weights[name] = array

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = build_tf_xlnet_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        logger.info("Importing {}".format(name))
        if name not in tf_weights:
            logger.info("{} not in tf pre-trained weights, skipping".format(name))
            continue
        array = tf_weights[name]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if 'kernel' in name and ('ff' in name or 'summary' in name or 'logit' in name):
            logger.info("Transposing")
            array = np.transpose(array)

        if isinstance(pointer, list):
            # Here we will split the TF weigths
            assert len(pointer) == array.shape[0]
            for i, p_i in enumerate(pointer):
                arr_i = array[i, ...]
                try:
                    assert p_i.shape == arr_i.shape
                except AssertionError as e:
                    e.args += (p_i.shape, arr_i.shape)
                    raise
                logger.info("Initialize PyTorch weight {} for layer {}".format(name, i))
                p_i.data = torch.from_numpy(arr_i)

        else:
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/Adam', None)
        tf_weights.pop(name + '/Adam_1', None)
    logger.info("Weights not copied to PyTorch model: {}".format(', '.join(tf_weights.keys())))
    return model