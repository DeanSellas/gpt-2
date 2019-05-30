import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder


import time, random



class GPT2():
    def __init__(self, model_name='117M', seed=None, nsamples=1, batch_size=1, length=None, temperature=0.5, top_k=40):
        """
        GPT-2 class
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
        self.model_name = model_name
        self.seed = seed
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.length = length
        self.temperature = temperature
        self.top_k = top_k
        
        # builds encoder for model.
        self.enc = encoder.get_encoder(self.model_name)

        #builds paramerters to work off of
        self.hparams = model.default_hparams()
        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))
        
        self._checks()

        # if checks pass, start tensorflow
        self.sess = tf.Session()

        self._buildOutput()
        


    def _checks(self):
        """ Makes sure all values are appropriate values are inputted and corrects those values """
        if self.seed is None:
            # sets a random seed if seed is not defined
            self.seed = random.randint(0, 2**32-1)
            print("Your Seed is: " + str(self.seed))
        
        # makes sure batch size is divisable by samples. prevents errors when building outputs
        if self.batch_size is None or self.nsamples % self.batch_size == 0:
            # set batchsize to a default of 1
            self.batch_size = 1

        # makes sure the output can calculate the desired length
        if self.length is None:
            # default length = hparmas.n_ctx (max size) / 2
            self.length = self.hparams.n_ctx // 2
        
        elif self.length > self.hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.hparams.n_ctx)

    def _buildOutput(self):
        """ Builds the necessary objects for GPT-2 to run properly """
        
        # placeholder value, dtype, shape. Shape is batchzise and none
        self.context = tf.placeholder(tf.int32, [self.batch_size, None])
        print("Context Pre-Output: "+str(self.context))


        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.output = sample.sample_sequence(
            hparams = self.hparams, length = self.length,
            context = self.context,
            batch_size = self.batch_size,
            temperature = self.temperature, top_k = self.top_k
        )
        print("Output: "+str(self.output))

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
        saver.restore(self.sess, ckpt)


    def run(self, timer = False):
        """ Runs GPT-2 """

        print("Type ?help for available commands")
        while True:
            
            self._prompt()
            
            if self.raw_text == None:
                continue
            
            context_tokens = self.enc.encode(self.raw_text)
            generated = 0

            if timer:
                start = time.time()
            for _ in range(self.nsamples // self.batch_size):
                
                # Returns shape of the output
                # print("Output Tensor: "+str(output))
                
                # feed dictionary, consiting of the context tokens
                contextDic = {self.context: [context_tokens for placeholder in range(self.batch_size)]}

                # print("Context Dictionary: "+ str(contextDic))

                # TODO figure out how this function works
                #  OUT CREATES ALL THE OUTPUTS FOR THE GIVEN PROMPT
                # https://www.tensorflow.org/api_docs/python/tf/Session#run
                out = self.sess.run(self.output, feed_dict=contextDic)

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

                for i in range(self.batch_size):
                    generated += 1
                    text = self.enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)

                    # PRINTS LIST OF INDECIES
                    # print(out[i])
                    print(text)

            print("=" * 80)
            if timer:
                end = time.time()
                print("It took "+str(round(end - start)) + " seconds to generate this output")

        # ends program
        self.close()


    # Input for GPT-2 to customize settings
    def _prompt(self):
        "Takes user input and throws appropriate respose to GPT-2"

        self.raw_text = input("Model prompt >>> ")

        # if prompt is empty notify user, and restart loop
        if not self.raw_text:
            print("Prompt should not be empty! Type ?help or ?h for available commands")
            return None
        
        if self.raw_text == "?help" or self.raw_text == "?h":
            print("Available Commands:\n\n?help - Displays Commands. Alias: ?h\n#kill - Ends GPT-2 Process\n")
            return None

        # if user types #kill break the loop. Cleaner way to close application
        if self.raw_text == "#kill":
            check = input("Are you sure you want to end GPT-2? [Y/n] ").lower()
            if check == "n":
                return None
            self.close()

    def close(self):
        print("Ending GPT-2")
        self.sess.close()
        exit()




model_name='117M'
# made seed 20 to get same result every time. meant for testing
seed=None

# NSamples are how many outputs are generated
nsamples=2
# Batch Size is how many outputs are generated at once
batch_size=1

length=100
temperature=0.5
top_k=40

gpt = GPT2(model_name, seed, nsamples, batch_size, length, temperature, top_k)
gpt.run(True)