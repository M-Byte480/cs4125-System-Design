import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def trans_to_en(texts):
    t2t_m = "facebook/m2m100_418M"
    t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                 download_method=DownloadMethod.REUSE_RESOURCES)

    text_en_l = []
    for text in texts:
        if text == "":
            text_en_l = text_en_l + [text]
            continue

        doc = nlp_stanza(text)
        print(doc.lang)
        if doc.lang == "en":
            text_en_l = text_en_l + [text]
        else:
            lang = doc.lang
            if lang == "fro":  # fro = Old French
                lang = "fr"
            elif lang == "la":  # latin
                lang = "it"
            elif lang == "nn":  # Norwegian (Nynorsk)
                lang = "no"
            elif lang == "kmr":  # Kurmanji
                lang = "tr"

            case = 2

            if case == 1:
                text_en = t2t_pipe(text, forced_bos_token_id=t2t_pipe.tokenizer.get_lang_id(lang='en'))
                text_en = text_en[0]['generated_text']
            elif case == 2:
                tokenizer.src_lang = lang
                encoded_hi = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
                text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                text_en = text_en[0]
            else:
                text_en = text

            text_en_l = text_en_l + [text_en]

            print(text)
            print(text_en)

    return text_en_l
	
#Calling translation method
# Note that the we can only translate a limited number of words so we are only translating ticket summary and not interaction content

temp["ts_en"] = trans_to_en(temp["ts"].to_list())