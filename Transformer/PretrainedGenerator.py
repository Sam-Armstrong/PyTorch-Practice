from transformers import pipeline

text_generation = pipeline('text-generation', 'gpt2')

prefix_text = "The world is"

generated_text = text_generation(prefix_text, max_length = 50, do_sample = False)[0]
print(generated_text['generated_text'])