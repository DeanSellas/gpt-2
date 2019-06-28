import json, os, time, random
import numpy as np
import tensorflow as tf

import sys
from datetime import datetime

from pyLogger import pyLogger

import model, sample, encoder

class GPT2():
    """
        GPT-2 class
        ------------------------------------------------------------------------------------------------
        :model_name=117M : String, which model to use

        :seed= -1 : Integer seed for random number generators, fix seed to reproduce
        results. If seed <= -1 (default) then generate a random seed between 0 and 2^32 - 1

        :nsamples=1 : Number of samples to return total

        :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.

        :length=0 : Number of tokens in generated text, if 0 (default), is
        determined by model hyperparameters

        :temperature=1 : Float value controlling randomness in boltzmann
        distribution. Lower temperature results in less random completions. As the
        temperature approaches zero, the model will become deterministic and
        repetitive. Higher temperature results in more random completions.

        :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
        considered for each step (token), resulting in deterministic completions,
        while 40 means 40 words are considered at each step. 0 (default) is a
        special setting meaning no restrictions. 40 generally is a good value.

        :log_path : Path to log file. If one is not provided the default path will be used (logs/log.txt)

        :debug : Puts GPT-2 into a debugging mode. This allows for more logging and other useful features 
        ------------------------------------------------------------------------------------------------
    """

    def __init__(self, model_name='117M', seed=-1, nsamples=1, batch_size=1, length=0, temperature=1, top_k=0, log_path=None, debug=False):
        self.model_name = model_name
        self.seed = seed
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.length = length
        self.temperature = temperature
        self.top_k = top_k
        self.log_path = log_path
        self.debug = debug

        
        
        # builds encoder for model.
        self.enc = encoder.get_encoder(self.model_name)

        #builds paramerters to work off of
        self.hparams = model.default_hparams()
        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        # if checks pass, start tensorflow
        self.sess = tf.Session()

        self._getReady()


    def _getReady(self):
        """
            Builds the necessary objects for GPT-2 to run properly
        """
        # starts logger
        self.logger = pyLogger(self.log_path)
        self.logger._print("STARTING GPT-2 -- {}\n\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), None)
        
        # checks properties before building application
        self._checks()

        # placeholder value, dtype, shape. Shape is batchzise and none
        self.context = tf.placeholder(tf.int32, [self.batch_size, None])
        self.logger._print("Context Pre-Output: {}".format(self.context))


        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        # builds the sample for tensorflow
        self.buildSeq()

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
        saver.restore(self.sess, ckpt)

        self.logger._print("Started GTP-2 With Params -- Seed: {}; Length: {}; nSamples: {}; Batch-Size: {}; Top-K: {}; Tempurature: {}".format(self.seed, self.length, self.nsamples, self.batch_size, self.top_k, self.temperature))

    def buildSeq(self):
        # generates a sample set sequence of data based on parameters provided
        self.sequence = sample.sample_sequence(
            hparams = self.hparams, length = self.length,
            context = self.context,
            batch_size = self.batch_size,
            temperature = self.temperature, top_k = self.top_k
        )

        # saves that data inside self.sequence
        # self.logger._print("Output: "+str(self.sequence))

    def _checks(self):
        """
            Makes sure all values are appropriate values are inputted and corrects those values
        """

        # checks the current seed and if it is not assigned random seed is generated
        if self.seed < 1 or type(self.seed) is not int:
            self.seed = random.randint(0, 2**32-1)
            self.logger._print("Seed was not defined. Your Seed is: " + str(self.seed), 'W')

        if self.nsamples < 1:
            self.logger._print("nSamples must be an integer greater than 0. nSamples set to 1", 'W')
            self.nsamples = 1
        
        # makes sure batch size is divisable by samples. prevents errors when building outputs
        if self.nsamples % self.batch_size == 0:
            # set batchsize to a default of 1
            self.logger._print("Batch Size is too large for the amount of samples provided. Batch Size was set to 1.", 'W')
            self.batch_size = 1

        # makes sure the output can calculate the desired length
        if self.length < 1 or self.length > self.hparams.n_ctx:
            # default length = hparmas.n_ctx (max size) / 2
            self.length = self.hparams.n_ctx // 2
            self.logger._print("Length was not defined or too long. Length was set to {}".format(self.length), 'W')

        if self.temperature < 1:
            self.logger._print("Temperature must be 1 or higher. Temperature was set to 1", 'W')
            self.temperature = 1

        if self.top_k < 0:
            self.logger._print("Top_k must be 0 or higher. Top_k was set to 0", 'W')
            self.top_k = 0

        # checks the log path assigned. If path does not exist, use default path
        if self.log_path != None and os.path.exists(self.log_path):
            self.log_path = None
            self.logger.path = self.log_path
            self.logger._print("Path provided for the logs does not exist! Path has been set to log folder", 'W')


    def run(self):
        """
            Runs GPT-2

            Debug allows the user to see more output. Also displays time it took for the program to create and display the output
        """

        self.logger._print("Type ?help for available commands")
        while True:
            self._prompt()
            
            if self.raw_text == None:
                continue
            
            # encodes the inputted string into a format that can be inputted into Tensor Flow
            context_tokens = self.enc.encode(self.raw_text)

            if self.debug:
                self.logger._print("Seed: {}; Length: {}; nSamples: {}; Batch-Size: {};".format(self.seed, self.length, self.nsamples, self.batch_size))
                start = time.time()
            
            for generated in range(self.nsamples // self.batch_size):
                
                # Returns shape of the output
                # print("Output Tensor: "+str(output))

                # feed dictionary, consiting of the context tokens
                # context tolkens are created by encoding the values provided by the user
                contextDic = {self.context: [context_tokens for placeholder in range(self.batch_size)]}

                # print("Context Dictionary: "+ str(contextDic))


                """
                    TODO figure out how this function works

                    OUT CREATES ALL THE OUTPUTS FOR THE GIVEN PROMPT
                    https://www.tensorflow.org/api_docs/python/tf/Session#run

                    Encoded text and model data is sent into Tensorflow.

                    Tensorflow is then able to create a output based on sample sequence and raw input.
                """
                out = self.sess.run(self.sequence, feed_dict=contextDic)

                # Removes the context tokens (inputted string) from the output.

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

                    -- i am unsure if i can get the Top-K. This is because it seems to be stored inside tensorflow and I have not found a way to output it -- 
                """

                for i in range(self.batch_size):
                    text = self.enc.decode(out[i])
                    self.logger._print("=" * 40 + " SAMPLE " + str(generated+1) + " " + "=" * 40, None)

                    # PRINTS LIST OF INDECIES
                    # print(out[i])
                    self.logger._print(text, "OUTPUT")

            print("=" * 80)
            if self.debug:
                end = time.time()
                self.logger._print("It took "+str(round(end - start)) + " seconds to generate this output")
            self.logger._save()

        # ends program
        self.close()


    # Input for GPT-2 to customize settings
    def _prompt(self):
        "Takes user input and throws appropriate respose to GPT-2"

        self.raw_text = input("Model prompt >>> ")

        # if prompt is empty notify user, and restart loop
        if not self.raw_text:
            print("Prompt should not be empty! Type ?help or ?h for available commands")
            self.raw_text = None
        
        elif self.raw_text == "?help" or self.raw_text == "?h":
            print("Available Commands:\n\n?help - Displays Commands. Alias: ?h\n#kill - Ends GPT-2 Process\n")
            self.raw_text = None

        # if user types #kill break the loop. Cleaner way to close application
        elif self.raw_text == "#kill":
            check = input("Are you sure you want to end GPT-2? [Y/n] ").lower()
            if check == "n":
                self.raw_text = None
            self.close()
        
        elif self.raw_text == "#change":
            print("What variable would you like to modify?\n\nnsamples: {} \nbatch_size: {} \nlength: {}".format(self.nsamples, self.batch_size, self.length))
            
            variable = input("Type the Variable: ").lower()
            try:
                value = int(input("New Value: "))

                if variable == "batch_size":
                    self.batch_size = value
                elif variable == "nsamples":
                    self.nsamples = value
                elif variable == "length":
                    self.length = value
                    self.buildSeq()
            except:
                self.logger._print("Please use an integer!", 'E')
            
            self.raw_text = None

    def close(self):
        self.logger._print("Ending GPT-2")
        self.logger._print("\n"+"="*80, None)
        
        self.logger.close()
        self.sess.close()
        sys.exit()
