from interact_new import GPT2
from fire import Fire

def interact_model(
    model_name='117M',
    # made seed 20 to get same result every time. meant for testing
    seed=-1,

    # NSamples are how many outputs are generated
    nsamples=1,
    # Batch Size is how many outputs are generated at once
    batch_size=1,

    length=0,
    temperature=1,
    top_k=0,
    log_path = None,
    debug = False,
):
    gpt = GPT2(model_name, seed, nsamples, batch_size, length, temperature, top_k, log_path, debug)
    gpt.run()

if __name__ == '__main__':
    Fire(interact_model)