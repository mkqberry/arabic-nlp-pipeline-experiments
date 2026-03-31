import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyspark.sql import SparkSession

spark = SparkSession.builder\
                    .appName("JAIS Deployment")\
                    .master("spark://10.155.123.139:8001")\
                    .config("spark.executor.instances", "2")\
                    .config("spark.executor.cores", "64")\
                    .config("spark.driver.memory", "115g")\
                    .config("spark.default.parallelism", "2")\
                    .config("spark.default.partitions", "2")\
                    .config("spark.executor.memory", "115g").getOrCreate()


MODEL_PATH = "/data_1/workspace/m00836648/jais/model/models--inception-mbzuai--jais-13b-chat/snapshots/2a47bcd25d5c7cc5a528ed86ebfe147480929c5d"
MODEL_PATH2 = "/opt/huawei/data/workspace/m00836648/jais/model/models--inception-mbzuai--jais-13b-chat/snapshots/2a47bcd25d5c7cc5a528ed86ebfe147480929c5d"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def get_response(text, tokenizer, model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(DEVICE)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=2048-input_len,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    response = response.split("### Response: [|AI|]")
    return response

def process_data_using_model(text):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH2, 
                                                 device_map="auto", 
                                                 trust_remote_code=True)

    input_ids = tokenizer(text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

data = [("Tell me a joke",), ("How are you?",)]
instruction = '### Instruction: You are highly skilled AI assistant. \n### Input: [|Human|] "{Input}" \n### Response: [|AI|]'
prompt1 = instruction.format_map({'Input': data[0][0]})
#promt2 = instruction.format_map({'Input': data[1][0]})

dataRDD = spark.sparkContext.parallelize([(prompt1,)])

processed_data = dataRDD.map(lambda text: process_data_using_model(text=text))

results = processed_data.collect()

for result in results:
    print(result)

spark.stop()
