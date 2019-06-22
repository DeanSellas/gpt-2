from interact_new import GPT2
from fire import Fire

def interact_model(
    model_name='117M',
    # made seed 20 to get same result every time. meant for testing
    seed=None,

    # NSamples are how many outputs are generated
    nsamples=2,
    # Batch Size is how many outputs are generated at once
    batch_size=2,

    length=None,
    temperature=0.5,
    top_k=40,
    debug = False,
    log_path = None,
):
    gpt = GPT2(model_name, seed, nsamples, batch_size, length, temperature, top_k, log_path, debug)
    gpt.run()

if __name__ == '__main__':
    Fire(interact_model)