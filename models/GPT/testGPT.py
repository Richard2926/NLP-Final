from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer

model = TFGPT2LMHeadModel.from_pretrained('./output', from_pt=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

inp = input("Enter a string to start of a joke for GPT-2 to continue, or leave this blank for GPT-2 to generate a random joke (Use 'q' to quit):")
while inp != 'q':
    input_ids = tokenizer.encode("<|joke|>" + inp, return_tensors='tf')

    generated_text_samples = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=5,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_p=0.92,
        temperature=.85,
        do_sample=True,
        top_k=125,
        early_stopping=True
    )

    for i, output in enumerate(generated_text_samples):
        print(tokenizer.decode(output, skip_special_tokens=True))
    print()
    inp = input("Joke Start: ")
