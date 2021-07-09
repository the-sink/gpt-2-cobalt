#!/usr/bin/env python3

import fire
import json
import os
import sys
import math
import numpy as np
import tensorflow.compat.v1 as tf
from http.server import HTTPServer, BaseHTTPRequestHandler

import model, sample, encoder

def interact_model(
    model_name='345M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=0.8,
    top_k=40,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        lTotal = math.floor(length / 3);

        output = sample.sample_sequence(
            hparams=hparams, length=lTotal,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        class Serv(BaseHTTPRequestHandler):
            def do_POST(self):
                try: 
                    message = self.rfile.read(int(self.headers.get('Content-Length'))).decode("utf-8")
                    print(message)
                    context_tokens = enc.encode(message)
                    generated = 0
                    response = ""

                    output = sample.sample_sequence(
                        hparams=hparams, length=lTotal,
                        context=context,
                        batch_size=batch_size,
                        temperature=temperature, top_k=top_k, top_p=top_p
                    )

                    for _ in range(nsamples // batch_size):
                        out = sess.run(output, feed_dict={
                            context: [context_tokens for _ in range(batch_size)]
                        })[:, len(context_tokens):]
                        for i in range(batch_size):
                            generated += 1
                            response = enc.decode(out[i])

                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(response.encode("utf8"))
                except:
                    print("err: ", sys.exc_info()[0])
                    self.send_response(500)
                    self.end_headers()


        httpd = HTTPServer(('0.0.0.0',7001),Serv)
        print("server running")
        httpd.serve_forever()

if __name__ == '__main__':
    fire.Fire(interact_model)
