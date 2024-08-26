# @Time : 2023/12/27 21:15
# @Author : Li Jiaqi
# @Description :
import random

from chatgpt import ModelEnums
from load_data import separate_citations
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

neg_words = ['not mentioned', 'not provided', 'not given', "'t mentioned", "'t provided", "'t given", 'not provide',
             "'t provide", "'t mention", "not mention", "no context provided", "insufficient"]


class PromptEnums:
    NORMAL = "normal"
    HYGIRD = "hygrid"
    HYGIRD_FINE_GRAINED = "hygrid_fine_grained"
    ADD_CITATION_DIRECTLY = "add_citation_directly"
    ADD_CITATION_IN_CONTEXT = "add_citation_in_context"
    ADD_CITATION_FINE_GRAINED = "add_citation_fine_grained"
    CATEGORIZE_CITATION_IN_CONTEXT = "categorize_citation_in_context"
    ORION_RAG_QA = "orion_rag_qa"
    LLAMA2_CHAT = "llama2_chat"
    VICUNA_CHAT = "vicuna_chat"
    SELF_RAG = "self_rag"
    QUERY_AUGMENTATION = "query_augmentation"
    TASK_COGNITION = "task_cognition"
    TASK_QA = "task_qa"
    TASK_PROMPT = "task_prompt"
    LLAMA2_VALIDATION = "llama2_validation"
    MISTRAL_CHAT = "mistral"
    LLAMA3_CHAT = "llama3_chat"
    GEMMA_CHAT="gemma_chat"


model_prompt_map = {
    ModelEnums.GPT3T: PromptEnums.NORMAL,
    ModelEnums.GPT4: PromptEnums.NORMAL,
    ModelEnums.GPT4T: PromptEnums.NORMAL,
    ModelEnums.ORION_RAG_QA_14B: PromptEnums.ORION_RAG_QA,
    ModelEnums.LOCAL: PromptEnums.NORMAL,
    ModelEnums.LLAMA2_CHAT_7B: PromptEnums.LLAMA2_CHAT,
    ModelEnums.VICUNA_7B: PromptEnums.VICUNA_CHAT,
    ModelEnums.SELF_RAG: PromptEnums.SELF_RAG,
    ModelEnums.COGNITION: PromptEnums.LLAMA2_CHAT,
    ModelEnums.COGNITION_QA: PromptEnums.LLAMA2_CHAT,
    ModelEnums.PROMPT_CENTERED: PromptEnums.LLAMA2_CHAT,
    ModelEnums.PROMPT_CENTERED_QA_COGNITION: PromptEnums.LLAMA2_CHAT,
    ModelEnums.QA_MODEL: PromptEnums.LLAMA2_CHAT,
    ModelEnums.PROMPT_CENTERED_QA: PromptEnums.LLAMA2_CHAT,
    ModelEnums.TEMP: PromptEnums.LLAMA3_CHAT,

    ModelEnums.MISTRAL: PromptEnums.MISTRAL_CHAT,
    ModelEnums.MISTRAL_PSQA: PromptEnums.MISTRAL_CHAT,
    ModelEnums.MISTRAL_COGNITION_QA:PromptEnums.MISTRAL_CHAT,
    ModelEnums.MISTRAL_COGNITION: PromptEnums.MISTRAL_CHAT,

    ModelEnums.MAMBA: PromptEnums.NORMAL,

    ModelEnums.GEMMA: PromptEnums.GEMMA_CHAT,
    ModelEnums.GEMMA_PSQA: PromptEnums.GEMMA_CHAT,
    ModelEnums.GEMMA_COGNITION_QA: PromptEnums.GEMMA_CHAT,
    ModelEnums.GEMMA_COGNITION: PromptEnums.GEMMA_CHAT,

    ModelEnums.LLAMA3: PromptEnums.LLAMA3_CHAT,
    ModelEnums.LLAMA3_COGNITION: PromptEnums.LLAMA3_CHAT,
    ModelEnums.LLAMA3_COGNITION_QA: PromptEnums.LLAMA3_CHAT,
    ModelEnums.LLAMA3_PSQA: PromptEnums.LLAMA3_CHAT,
}


def get_task_prompt(context, query, version=PromptEnums.TASK_COGNITION, specific_instruction=""):
    VERSIONS = [PromptEnums.TASK_COGNITION, PromptEnums.QUERY_AUGMENTATION, PromptEnums.TASK_QA,
                PromptEnums.TASK_PROMPT]
    assert version in VERSIONS, f"version must be one of {VERSIONS}"

    assert_instructions = ["", "", "", "", ""]
    assert_instructions[random.randint(0, len(assert_instructions) - 1)] = specific_instruction

    res = ""
    if version == PromptEnums.TASK_COGNITION:
        res = (
            "Cognition Assessment Task: You need to do the Cognition Assessment Task for the following query and context. "
            "I will give a query and a related context about the query. "
            "Your task is to judge whether the context is sufficient to answer the query. "
            "You must append either '<Sufficient>' or '<Insufficient>' after your answer. "
            "Your answer must not using any additional knowledge that is not mentioned in the given contexts."
            "If the context is not sufficient to answer the question, please append the '<Insufficient>' after your answer.\n\n"
            "Here is the example.\n\n QUERY:\n What happened to Jay Chou when he got old?\n"
            "CONTEXT:\n Jay Chou was the most famous singer in China when he was young, "
            "releasing many nostalgic albums and songs that are memorable to middle-aged people today.\n"
            "ANSWER: \nJay Chou was the most famous singer in China.<Insufficient>\n\n"
            "Here is one query that you need to accomplish, you should only output one answer to it, and stop generating more queries:\n\n"
            f"QUERY:\n {query}\n"
            f"CONTEXT:\n {context}\n"
            "ANSWER: \n"
        )
    elif version == PromptEnums.QUERY_AUGMENTATION:
        res = (
            "I will give a positive query and a related context about the query. "
            "Your task is to generate a negative query that is related to the topic of the context, "
            "but cannot find an answer from the context. \n\n"
            "e.g. POSITIVE QUERY:\n What happened to Jay Chou when he was young?\n"
            "CONTEXT:\n Jay Chou was the most famous singer in China when he was young, "
            "releasing many nostalgic albums and songs that are memorable to middle-aged people today.\n"
            "NEGATIVE QUERY: \nWhat happened to Jay Chou when he got old?\n\n"
            "Here is the provided information that you need to accomplish follow the provided example:\n\n"
            f"POSITIVE QUERY:\n {query}\n"
            f"CONTEXT:\n {context}\n"
            "NEGATIVE QUERY: \n"
        )
    elif version == PromptEnums.TASK_QA:
        res = (
            f"Question Answering Task: You need to do the Question Answering Tasks for the following query and context. {assert_instructions[0]}\n"
            f"I will give a question and several context texts about the question. {assert_instructions[1]}"
            f"Based on the given contexts, give an answer to the question. {assert_instructions[2]}"
            f"Your answer must not using any additional knowledge that is not mentioned in the given contexts. {assert_instructions[3]}"
            f"If the context is not sufficient to answer the question, please answer it with 'Not Provided' {assert_instructions[4]}\n\n"
            "QUESTION:\n"
            f"{query}\n\n"
            "CONTEXTS:\n"
            f"{context}\n\n"
            "ANSWER: \n"
        )
    elif version == PromptEnums.TASK_PROMPT:
        res = (
            "You should check whether your answer aligned the requirement by generating a Checking part, checking each sentence of the above instruction, "
            "with either <fulfilled> or <not fulfilled> mark behind the sentence, indicating whether the requirement is fulfilled or not.\n"
            "If there is <not fulfilled> mark behind the sentence, you must modify your answer again to fulfill the requirement, by appending a new ANSWER and CHECKING part.\n\n"
            "e.g. Question Answering Task Requirements: You need to do the Task Prompt for the following query and context.\nEnsure the response is written in the past tense.\n\n"
            "QUESTION:\n Who is Jack Chen?\n\nCONTEXTS:\n People saying that Jack Chen is a famous singer in China. \n\nANSWER: Jack Chen was a famous singer in China.\n\n "
            "CHECKING:\nQuestion Answering Task: You need to do the Task Prompt for the following query and context.<fulfilled>\nEnsure the response is written in the past tense.<fulfilled>\n\n\n"


            "Here is an example for this task:\n\n"
            "e.g. Question Answering Task Requirements: You need to do the Task Prompt for the following query and context.\nEnsure the response is written in the past tense.\n\n"
            "QUESTION:\n Who is Jack Chen?\n\nCONTEXTS:\n People saying that Jack Chen is a famous singer in China. \n\nANSWER: Jack Chen is a famous singer in China.\n "
            "CHECKING:\nQuestion Answering Task: You need to do the Task Prompt for the following query and context.<fulfilled>\nEnsure the response is written in the past tense.<not fulfilled>\n"
            "ANSWER: Jack Chen was a famous singer in China.\n"
            "CHECKING:\nQuestion Answering Task: You need to do the Task Prompt for the following query and context.<fulfilled>\nEnsure the response is written in the past tense.<fulfilled>\n\n\n"
            "Here is the information of your task:\n\n"
            f"Question Answering Task Requirements: You need to do the Task Prompt for the following query and context. {assert_instructions[0]}\n"
            f"I will give a question and several context texts about the question. {assert_instructions[1]}"
            f"Based on the given contexts, give an answer to the question. {assert_instructions[2]}"
            f"Your answer must not using any additional knowledge that is not mentioned in the given contexts. {assert_instructions[3]}"
            f"If the context is not sufficient to answer the question, please answer it with 'Not Provided' {assert_instructions[4]}\n\n"
            "QUESTION:\n"
            f"{query}\n\n"
            "CONTEXTS:\n"
            f"{context}\n\n"
            "ANSWER: \n"
        )
    return res


def get_citation_prompt(question, context, answer=None, task_assignment="",
                        version=PromptEnums.ADD_CITATION_DIRECTLY):
    VERSIONS = [
        PromptEnums.NORMAL, PromptEnums.HYGIRD, PromptEnums.HYGIRD_FINE_GRAINED,
        PromptEnums.ADD_CITATION_DIRECTLY,
        PromptEnums.ADD_CITATION_IN_CONTEXT,
        PromptEnums.ADD_CITATION_FINE_GRAINED,
        PromptEnums.CATEGORIZE_CITATION_IN_CONTEXT,
        PromptEnums.ORION_RAG_QA,
        PromptEnums.LLAMA2_CHAT,
        PromptEnums.VICUNA_CHAT,
        PromptEnums.SELF_RAG,
        PromptEnums.TASK_PROMPT,
        PromptEnums.LLAMA2_VALIDATION,
        PromptEnums.MISTRAL_CHAT,
        PromptEnums.LLAMA3_CHAT,
        PromptEnums.GEMMA_CHAT
    ]
    assert version in VERSIONS, f"version must be one of {VERSIONS}"
    if version == PromptEnums.NORMAL:
        return (f"{task_assignment}"
                "I will give a question and several context texts about the question. "
                "Based on the given contexts, give a informative answer to the question. "
                "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
                "If the context is not sufficient to answer the question, please answer it with 'Not Provided'\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "CONTEXTS:\n"
                f"{context}\n\n"
                "ANSWER: \n"
                )
    elif version == PromptEnums.HYGIRD:
        return (f"{task_assignment}"
                "I will give a question and several context texts about the question. "
                "Based on the given contexts, give a informative answer to the question. "
                "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
                "Also your answer must be true in the real world without any hallucination. "
                "You can list some points or draw a table based on markdown syntax to illustrate your idea. "
                "Also, you must mention the reference of parts of your answer based on the "
                "given contexts within brackets [] as in the IEEE format seperated by ',' if there are multiple references.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "CONTEXTS:\n"
                f"{context}\n\n"
                "ANSWER: \n"
                )
    elif version == PromptEnums.ADD_CITATION_DIRECTLY:
        return (f"{task_assignment}"
                "I will give a question, several context texts started with reference number, and an answer without citation about the question. "
                "You should provide an answer with citations that must keeps all the content without modification from the provided answer, while adding the citations into the answer. "
                "You need to mention the reference of parts of the answer based on the "
                "given contexts within brackets [] as in the IEEE format seperated by ',' if there are multiple references.\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "CONTEXTS:\n"
                f"{context}\n\n"
                f"ANSWER without CITATION: \n{answer}\n\n"
                "ANSWER within CITATION: \n"
                )
    elif version == PromptEnums.ADD_CITATION_IN_CONTEXT:
        in_context_query, in_context_context, in_context_answer_without_citation, in_context_answer_with_citation = get_in_context_learning_sample()
        return (f"{task_assignment}"
                "I will give a question, several context texts started with reference number, and an answer without citation about the question. "
                "You should provide an answer with citations that must keeps all the content from the answer without citation, while adding the citations into the answer. "
                "You need to mention the reference of parts of the answer based on the "
                "given contexts within brackets [] as in the IEEE format seperated by ',' if there are multiple references.\n\n"
                "Here is an example:\n"
                "QUESTION: Are there any significant changes in properties from 2019 10K reporting period to another?\n"
                f"{in_context_query}\n\n"
                "CONTEXTS:\n"
                f"{in_context_context}\n\n"
                f"ANSWER without CITATION: \n{in_context_answer_without_citation}\n\n"
                f"ANSWER with CITATION: \n{in_context_answer_with_citation}\n\n"

                "Here is the provided information that you need to accomplish follow the provided example:\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "CONTEXTS:\n"
                f"{context}\n\n"
                f"ANSWER without CITATION: \n{answer}\n\n"
                "ANSWER within CITATION: \n"
                )
    elif version == PromptEnums.HYGIRD_FINE_GRAINED:
        in_context_query, in_context_context, in_context_answer_without_citation, in_context_answer_with_citation = get_in_context_learning_sample(
            fine_grained=True)
        return (f"{task_assignment}""I will give a question and several context texts about the question. "
                "Based on the given contexts, give a informative answer to the question. "
                "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
                "Also your answer must be true in the real world without any hallucination. "
                "You can list some points or draw a table based on markdown syntax to illustrate your idea. "
                "Also, you must mention the reference of parts of your answer based on the "
                "given contexts within brackets [] as in the IEEE format seperated by ',' if there are multiple references, "
                "and mention the key related parts of the original contexts based on your reference within <> following the reference, e.g. [reference_number]<key_parts_of_the_original_contexts>\n\n"
                "Here is an example:\n"
                "QUESTION: Are there any significant changes in properties from 2019 10K reporting period to another?\n"
                f"{in_context_query}\n\n"
                "CONTEXTS:\n"
                f"{in_context_context}\n\n"
                f"ANSWER: \n{in_context_answer_with_citation}\n\n"

                "Here is the provided information that you need to accomplish follow the provided example:\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "CONTEXTS:\n"
                f"{context}\n\n"
                "ANSWER: \n"
                )
    elif version == PromptEnums.ADD_CITATION_FINE_GRAINED:
        in_context_query, in_context_context, in_context_answer_without_citation, in_context_answer_with_citation = get_in_context_learning_sample(
            fine_grained=True)
        return (f"{task_assignment}"
                "I will give a question, several context texts started with reference number, and an answer without citation about the question. "
                "You should provide an answer with citations that must keeps all the content from the answer without citation, while adding the citations into the answer. "
                "You need to mention the reference of parts of the answer based on the "
                "given contexts within brackets [] as in the IEEE format seperated by ',' if there are multiple references, "
                "and mention the key related parts of the original contexts based on your reference within <> following the reference, e.g. [reference_numbers]<key_parts_of_the_original_contexts>\n\n"
                "Here is an example:\n"
                "QUESTION: Are there any significant changes in properties from 2019 10K reporting period to another?\n"
                f"{in_context_query}\n\n"
                "CONTEXTS:\n"
                f"{in_context_context}\n\n"
                f"ANSWER without CITATION: \n{in_context_answer_without_citation}\n\n"
                f"ANSWER with CITATION: \n{in_context_answer_with_citation}\n\n"

                "Here is the provided information that you need to accomplish follow the provided example:\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "CONTEXTS:\n"
                f"{context}\n\n"
                f"ANSWER without CITATION: \n{answer}\n\n"
                "ANSWER within CITATION: \n"
                )
    elif version == PromptEnums.CATEGORIZE_CITATION_IN_CONTEXT:
        in_context_query, in_context_context, in_context_answer_without_citation, in_context_answer_with_citation = get_in_context_learning_sample(
            fine_grained=True, with_category=True)
        return (f"{task_assignment}"
                "I will give a question, several context texts started with reference number, and an answer without citation about the question. "
                "You should provide an answer with citations that must keeps all the content from the answer without citation, while adding the citations into the answer. "
                "You need to mention the reference of parts of the answer based on the "
                "given contexts within brackets [] as in the IEEE format seperated by ',' if there are multiple references, "
                "and mention the category of this citation and the key related parts of the original contexts based on your reference within <> following the reference, e.g. [reference_numbers]<category: key_parts_of_the_original_contexts>\n\n"

                "The categories of the citation are as follows:\n"
                "Quotes: If you are directly quoting from a source, you need to include a citation. \n"
                "Paraphrases: If you are rephrasing or summarizing information from a source, you still need to include a citation. Even though you are not using the exact words, the contents are still coming from the original source.\n"
                "Data: If you are using number directly from a source, you need to include a citation. \n"
                "Ideas: If you are discussing someone else's ideas or theories, you need to include a citation. This gives credit to the original thinker and allows the reader to explore the idea or theory in more depth if they wish.\n"
                "Sources: If you are using an image, chart, or graph from a source, you need to include a citation. This gives credit to the original creator and allows the reader to find the original image if needed.\n\n"

                "Here is an example:\n"
                "QUESTION: Are there any significant changes in properties from 2019 10K reporting period to another?\n"
                f"{in_context_query}\n\n"
                "CONTEXTS:\n"
                f"{in_context_context}\n\n"
                f"ANSWER without CITATION: \n{in_context_answer_without_citation}\n\n"
                f"ANSWER with CITATION: \n{in_context_answer_with_citation}\n\n"

                "Here is the provided information that you need to accomplish follow the provided example:\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "CONTEXTS:\n"
                f"{context}\n\n"
                f"ANSWER without CITATION: \n{answer}\n\n"
                "ANSWER within CITATION: \n"
                )
    elif version == PromptEnums.ORION_RAG_QA:
        return (
            "你是由猎户星空开发的AI助手，你的名字叫聚言。你可以根据下面给出的参考资料和聊天历史来回答用户问题。\n\n"
            "### 参考资料 ###\n"
            f"{context}\n\n"
            "### 聊天历史 ###\n"
            '\"\"\n\n'
            "### 用户问题 ###\n"
            f"{question}\n\n"
            "### 回答要求 ###\n"
            "1. 你只能根据上面参考资料中给出的事实信息来回答用户问题，不要胡编乱造。\n"
            "2. 如果向用户提出澄清问题有助于回答问题，可以尝试提问。\n"
            "3. 如果参考资料中的信息不足以回答用户问题，请直接回答下面三个双引号中的内容：\"\"\"问题与上下文无关\"\"\"。\n"
            "4. 请你以第一人称并且用严谨的风格来回答问题，一定要用{language}来回答，并且基于事实详细阐述。\n"
        )
    elif version == PromptEnums.VICUNA_CHAT:
        return (f"{task_assignment}"
                "A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
                "### Human:\n"
                "I will give a question and several context texts about the question. "
                "Based on the given contexts, give a informative answer to the question. "
                "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
                "If the context is not sufficient to answer the question, please answer it with 'Not Provided'\n\n"
                "QUESTION:\n"
                f"{question}\n\n"
                "CONTEXTS:\n"
                f"{context}\n\n"
                "### Assistant:\n"
                )
    elif version == PromptEnums.LLAMA2_CHAT:
        return (
            "[INST] <<SYS>>\n"
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
            "<</SYS>>\n\n"
            "User:"
            f"{task_assignment}"
            "I will give a question and several context texts about the question. "
            "Based on the given contexts, give a informative answer to the question. "
            "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
            "If the context is not sufficient to answer the question, please answer it with 'Not Provided'\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "CONTEXTS:\n"
            f"{context}\n\n"
            "ANSWER: \n"
            " [/INST]"
        )
    elif version == PromptEnums.LLAMA3_CHAT:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
            "<|eot_id|>\n\n"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{task_assignment}"
            "I will give a question and several context texts about the question. "
            "Based on the given contexts, give a informative answer to the question. "
            "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
            "If the context is not sufficient to answer the question, please answer it with 'Not Provided'\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "CONTEXTS:\n"
            f"{context}\n\n"
            "ANSWER: \n"
            "<|eot_id|>\n\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif version == PromptEnums.GEMMA_CHAT:
        return (
            "<start_of_turn>user\n"
            "A chat between a curious human and an artificial intelligence assistant. "
            "You should give helpful, detailed, and polite answers to my questions.\n"
            f"{task_assignment}"
            "I will give a question and several context texts about the question. "
            "Based on the given contexts, give a informative answer to the question. "
            "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
            "If the context is not sufficient to answer the question, please answer it with 'Not Provided'\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "CONTEXTS:\n"
            f"{context}\n\n"
            "ANSWER: \n"
            "<start_of_turn>model\n"
        )
    elif version == PromptEnums.SELF_RAG:
        return (
            "### Instruction:\n"
            f"{task_assignment}"
            "I will give a question and several context texts about the question. "
            "Based on the given contexts, give a informative answer to the question. "
            "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
            "If the context is not sufficient to answer the question, please answer it with 'Not Provided'\n\n"
            "question:"
            f"{question}\n\n"
            "### Response:\n"
            f"[Retrieval]<paragraph>{context}</paragraph>"
        )
    elif version == PromptEnums.TASK_PROMPT:
        return (
            "You should check whether your answer aligned the requirement by generating a Checking part, checking each sentence of the above instruction, "
            "with either <fulfilled> or <not fulfilled> mark behind the sentence, indicating whether the requirement is fulfilled or not.\n"
            "e.g. Question Answering Task Requirements: You need to do the Task Prompt for the following query and context.\nEnsure the response is written in the past tense.\n\n"
            "QUESTION:\n Who is Jack Chen?\n\nCONTEXTS:\n People saying that Jack Chen is a famous singer in China. \n\nANSWER: Jack Chen was a famous singer in China.\n\n "
            "CHECKING:\nQuestion Answering Task: You need to do the Task Prompt for the following query and context.<fulfilled>\nEnsure the response is written in the past tense.<fulfilled>\n"
            "If there is <not fulfilled> mark behind the sentence, you must modify your answer again to fulfill the requirement, by appending a new ANSWER and CHECKING part.\n"
            "e.g. Question Answering Task Requirements: You need to do the Task Prompt for the following query and context.\nEnsure the response is written in the past tense.\n\n"
            "QUESTION:\n Who is Jack Chen?\n\nCONTEXTS:\n People saying that Jack Chen is a famous singer in China. \n\nANSWER: Jack Chen is a famous singer in China.\n "
            "CHECKING:\nQuestion Answering Task: You need to do the Task Prompt for the following query and context.<fulfilled>\nEnsure the response is written in the past tense.<not fulfilled>\n"
            "ANSWER: Jack Chen was a famous singer in China.\n"
            "CHECKING:\nQuestion Answering Task: You need to do the Task Prompt for the following query and context.<fulfilled>\nEnsure the response is written in the past tense.<fulfilled>\n\n"
            "Here is the information of your task:\n"
            f"Question Answering Task Requirements: You need to do the Task Prompt for the following query and context. \n"
            f"I will give a question and several context texts about the question. "
            f"Based on the given contexts, give an answer to the question. "
            f"Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
            f"If the context is not sufficient to answer the question, please answer it with 'Not Provided' \n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "CONTEXTS:\n"
            f"{context}\n\n"
            "ANSWER: \n"
        )
    elif version == PromptEnums.LLAMA2_VALIDATION:
        return (
            "[INST] <<SYS>>\n"
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
            "<</SYS>>\n\n"
            "User:"
            f"{task_assignment}"
            "I will give a question and several context texts about the question. "
            "Based on the given contexts, give a informative answer to the question. "
            "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
            "If the context is not sufficient to answer the question, please answer it with 'Not Provided'"
            "An example answer is also provided. "
            "However, the answer could be wrong, you should identify the incorrect parts and ensure the correctness of your answer.\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "CONTEXTS:\n"
            f"{context}\n\n"
            "EXAMPLE ANSWER: \n"
            f"{answer}\n\n"
            "CORRECT ANSWER: \n"
            " [/INST]"
        )
    elif version == PromptEnums.MISTRAL_CHAT:
        return (
            "<s>[INST] "
            f"{task_assignment}"
            "I will give a question and several context texts about the question. "
            "Based on the given contexts, give a informative answer to the question. "
            "Your answer must not using any additional knowledge that is not mentioned in the given contexts. "
            "If the context is not sufficient to answer the question, please answer it with 'Not Provided'\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "CONTEXTS:\n"
            f"{context}\n\n"
            "ANSWER: \n"
            " [/INST]"
        )


# Quotes: If you are directly quoting from a source, you need to include a citation. This citation should include the page number (if available) to allow the reader to find the exact quote in the original source.
# Example: "We believe that our properties are in good condition, are well maintained and are suitable and adequate to carry on our business"[2]<Direct Quote>.
#
# Paraphrases: If you are rephrasing or summarizing information from a source, you still need to include a citation. Even though you are not using the exact words, the ideas are still coming from the original source.
# Example: In the 2020 report, the company provides a detailed list of their properties, including their locations, sizes, and whether they are owned or leased[2]<Paraphrases>.
#
# Data: If you are using data or statistics from a source, you need to include a citation. This allows the reader to verify the data if needed.
# Example: The company's corporate headquarters in Melville, NY is 185,000 square feet and is leased until July 2036[2]<Statistics>.
#
# Ideas: If you are discussing someone else's ideas or theories, you need to include a citation. This gives credit to the original thinker and allows the reader to explore the idea or theory in more depth if they wish.
# Example: The company believes that their properties are in good condition and are suitable for their business[2].
#
# Sources: If you are using an image, chart, or graph from a source, you need to include a citation. This gives credit to the original creator and allows the reader to find the original image if needed.
# Example: The chart showing the company's property locations and sizes is taken from their 2020 report[2].


def get_in_context_learning_sample(file_name='in-context-learning.txt', fine_grained=False, with_category=False):
    with open(file_name, 'r') as f:
        res = f.readlines()
        res = "\n".join(res)
        res = res.split('query:')[1]
        res = res.split('context:')
        query = res[0].strip()
        res = res[-1].split('answer_without_reference:')
        context = res[0].strip()
        res = res[-1].split('answer:')
        answer_without_citation = res[0].strip()
        answer_with_citation = res[-1].strip()
        if not fine_grained or not with_category:
            answer_with_citation_list = separate_citations(answer_with_citation)
            if not fine_grained and not with_category:
                for x in answer_with_citation_list:
                    x['citation'] = ""
            elif with_category:
                for x in answer_with_citation_list:
                    x['citation'] = "<" + ";".join(list(set([y.split(': ')[0] for y in x['citation']]))) + ">" if len(
                        x['citation']) != 0 else ""
            elif fine_grained:
                for x in answer_with_citation_list:
                    citation = []
                    for y in x['citation']:
                        citation.extend(y.split(': ')[1:])
                    x['citation'] = "<" + ";".join(list(set(citation))) + ">"
            answer_with_citation_list = [x['answer'] + x['citation'] for x in answer_with_citation_list]
            answer_with_citation = "".join(answer_with_citation_list)
    return query, context, answer_without_citation, answer_with_citation
