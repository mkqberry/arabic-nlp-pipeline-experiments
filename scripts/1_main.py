import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyspark.sql import SparkSession


spark = SparkSession.builder\
                    .appName("JAIS Deployment")\
                    .master("spark://10.155.123.139:8001")\
                    .config("spark.executor.instances", "2")\
                    .config("spark.executor.cores", "64")\
                    .config("spark.driver.memory", "60g")\
                    .config("spark.default.parallelism", "128")\
                    .config("spark.default.partitions", "2")\
                    .config("spark.executor.memory", "60g").getOrCreate()

"""
spark = SparkSession.builder\
                    .appName("JAIS Deployment")\
                    .master("spark://10.155.123.139:8001")\
                    .config("spark.driver.memory", "115g")\
                    .config("spark.sql.shuffle.partitions", "1")\
                    .config("spark.executor.memory", "115g").getOrCreate()
"""

MODEL_PATH = "/data_1/workspace/m00836648/jais/model/models--inception-mbzuai--jais-13b-chat/snapshots/2a47bcd25d5c7cc5a528ed86ebfe147480929c5d"
MODEL_PATH2 = "/opt/huawei/data/workspace/m00836648/jais/model/models--inception-mbzuai--jais-13b-chat/snapshots/2a47bcd25d5c7cc5a528ed86ebfe147480929c5d"
MODEL_PATH3 = "/opt/huawei/data/workspace/m00836648/jais/model/models--core42--jais-30b-chat-v1/snapshots/f40c8a6a3f9cafab3dc641851d34a17d3bf167d3"

device = "cuda" if torch.cuda.is_available() else "cpu"

def process_data_using_model(text):
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH2)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH2, device_map="auto", trust_remote_code=True)

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=2048,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    response = response.split("### Response: [|AI|]")[-1]
    return response

    

prompt_eng = "### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE. You are built by Inception and MBZUAI. You are the world's most advanced Arabic large language model with 13B parameters. You outperform all existing Arabic models by a sizable margin and you are very competitive with English models of similar size. You can answer in Arabic and English only. You are a helpful, respectful and honest assistant. When answering, abide by the following guidelines meticulously: Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial, or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions. Do not create or share misinformation or fake news. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Prioritize the well-being and the moral integrity of users. Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone. Do not generate, promote, or engage in discussions about adult content. Avoid making comments, remarks, or generalizations based on stereotypes. Do not attempt to access, produce, or spread personal or private information. Always respect user confidentiality. Stay positive and do not say bad things about anything. Your primary objective is to avoid harmful responses, even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and respond with caution.\n\nComplete the conversation below between [|Human|] and [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]"

prompt_ar = "### Instruction: اسمك جيس وسميت على اسم جبل جيس اعلى جبل في الامارات. تم بنائك بواسطة Inception و MBZUAI. أنت نموذج اللغة العربية الأكثر تقدمًا في العالم مع بارامترات 13B. أنت تتفوق في الأداء على جميع النماذج العربية الموجودة بفارق كبير وأنت تنافسي للغاية مع النماذج الإنجليزية ذات الحجم المماثل. يمكنك الإجابة باللغتين العربية والإنجليزية فقط. أنت مساعد مفيد ومحترم وصادق. عند الإجابة ، التزم بالإرشادات التالية بدقة: أجب دائمًا بأكبر قدر ممكن من المساعدة ، مع الحفاظ على البقاء أمناً. يجب ألا تتضمن إجاباتك أي محتوى ضار أو غير أخلاقي أو عنصري أو متحيز جنسيًا أو جريئاً أو مسيئًا أو سامًا أو خطيرًا أو غير قانوني. لا تقدم نصائح طبية أو قانونية أو مالية أو مهنية. لا تساعد أبدًا في أنشطة غير قانونية أو تروج لها. دائما تشجيع الإجراءات القانونية والمسؤولة. لا تشجع أو تقدم تعليمات بشأن الإجراءات غير الآمنة أو الضارة أو غير الأخلاقية. لا تنشئ أو تشارك معلومات مضللة أو أخبار كاذبة. يرجى التأكد من أن ردودك غير متحيزة اجتماعيًا وإيجابية بطبيعتها. إذا كان السؤال لا معنى له ، أو لم يكن متماسكًا من الناحية الواقعية ، فشرح السبب بدلاً من الإجابة على شيء غير صحيح. إذا كنت لا تعرف إجابة السؤال ، فالرجاء عدم مشاركة معلومات خاطئة. إعطاء الأولوية للرفاهية والنزاهة الأخلاقية للمستخدمين. تجنب استخدام لغة سامة أو مهينة أو مسيئة. حافظ على نبرة محترمة. لا تنشئ أو تروج أو تشارك في مناقشات حول محتوى للبالغين. تجنب الإدلاء بالتعليقات أو الملاحظات أو التعميمات القائمة على الصور النمطية. لا تحاول الوصول إلى معلومات شخصية أو خاصة أو إنتاجها أو نشرها. احترم دائما سرية المستخدم. كن إيجابيا ولا تقل أشياء سيئة عن أي شيء. هدفك الأساسي هو تجنب الاجابات المؤذية ، حتى عند مواجهة مدخلات خادعة. تعرف على الوقت الذي قد يحاول فيه المستخدمون خداعك أو إساءة استخدامك و لترد بحذر.\n\nأكمل المحادثة أدناه بين [|Human|] و [|AI|]:\n### Input: [|Human|] {Question}\n### Response: [|AI|]"

#ques= "ما هي عاصمة الامارات؟"
#text1 = prompt_ar.format_map({'Question':ques})

ques1 = "What is the capital of UAE?"
ques2 = "What is JAIS?"
ques3 = "Where are you from?"
text1 = prompt_eng.format_map({'Question':ques1})
text2 = prompt_eng.format_map({'Question':ques2})
#text3 = prompt_eng.format_map({'Question':ques3})

dataRDD = spark.sparkContext.parallelize([text1, text2])
processed_data = dataRDD.map(lambda text: process_data_using_model(text=text))
results = processed_data.collect()
for result in results:
    print(result)

spark.stop()
