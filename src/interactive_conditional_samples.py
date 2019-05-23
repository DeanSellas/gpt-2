#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

# Input for GPT-2 to customize settings
def prompt():
    "Takes user input and throws appropriate respose to GPT-2"

    raw_text = input("Model prompt >>> ")

    # if prompt is empty notify user, and restart loop
    if not raw_text:
        print("Prompt should not be empty! Type ?help or ?h for available commands")
        return None
    
    if raw_text == "?help" or raw_text == "?h":
        print("Available Commands:\n\n?help - Displays Commands. Alias: ?h\n#kill - Ends GPT-2 Process\n")
        return None

    # if user types #kill break the loop. Cleaner way to close application
    if raw_text == "#kill":
        check = input("Are you sure you want to end GPT-2? [Y/n] ").lower()
        if check == "n":
            return None
        print("Ending GPT-2")
        return "#kill"

    return raw_text


def interact_model(
    model_name='117M',
    # made seed 20 to get same result every time. meant for testing
    seed=20,

    # NSamples are how many outputs are generated
    nsamples=2,
    # Batch Size is how many outputs are generated at once
    batch_size=2,

    length=10,
    temperature=0.5,
    top_k=40,
):
    """
    Interactively run the model
    ------------------------------------------------------------------------------------------------
    :model_name=117M : String, which model to use

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
     ------------------------------------------------------------------------------------------------
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        print("Type ?help for available commands")
        while True:
            
            raw_text = prompt()
            
            if raw_text == None:
                continue
            
            if raw_text == "#kill":
                break
            
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                
                # Returns shape of the output
                # print("Output Tensor: "+str(output))
                
                # feed dictionary, consiting of the context tolkens
                contextDic = {context: [context_tokens for placeholder in range(batch_size)]}

                # print("Context Dictionary: "+ str(contextDic))

                # TODO figure out how this function works
                #  OUT CREATES ALL THE OUTPUTS FOR THE GIVEN PROMPT
                # https://www.tensorflow.org/api_docs/python/tf/Session#run
                out = sess.run(output, feed_dict=contextDic)

                # print(out)

                # Removes the context tokens from the output.

                # Context tokens are included in the output, this splices them out in order to give only the output and not the original prompt
                out = out[:, len(context_tokens):]

                # print(out)


                """
                ------------------------------------------------------------------------------------------------

                There seems to be a graph comparison going on here and returns a list with the indexes to the text.
                Out is a list of lists. Top list contains the sublists that hold the indices to the dictionary

                ------------------------------------------------------------------------------------------------

                What I believe is going on is, samples.py creates a graph that seems to be likely next words. This then gets thrown into the TensorFlow.run() function and is compaired to a dictionary. This then somehow comes up with an appropriate output to generate.

                ------------------------------------------------------------------------------------------------

                Because most of this seems to be handled inside of Tensorflow it may be difficult to gain access to the inner workings

                ------------------------------------------------------------------------------------------------

                TODO: see if there are other ways to generate Top-K.
                """

                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)

                    # PRINTS LIST OF INDECIES
                    print(out[i])
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)