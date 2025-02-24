
# https://rasa.com/docs/rasa/custom-actions

# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
#

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from typing import Text, Dict, List
import random
import sqlite3
from difflib import SequenceMatcher
import os
import pymongo
from pymongo import MongoClient
from langdetect import detect
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TranslationPipeline
import re
import subprocess
from dotenv import load_dotenv
from huggingface_hub import login
import time
from rasa.shared.core.events import FollowupAction

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["kazakh_learning_bot"]
users_collection = db["users"]
dictionary_collection = db["dictionary"]
quiz_results_collection = db["quiz_results"]
user_struggles_collection = db["user_struggles"]
vocab_quizzes_collection = db["vocab_quizzes"]
grammar_quizzes_collection = db["grammar_quizzes"]
mastering_collection = db["mastering_collection"]
knowledge_collection = db["knowledge_base"]


class ActionShowProgress(Action):
    def name(self) -> Text:
        return "action_show_learned"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain: Dict[Text, Any]) -> List[Dict]:
        user_id = "user123"

        mastering_data = mastering_collection.find_one({"user_id": user_id}) or {"words": []}
        learned_words = [word["word"] for word in mastering_data.get("words", []) if word["status"] == "learned"]
        learning_words = [word["word"] for word in mastering_data.get("words", []) if word["status"] == "learning"]

        response = "Your Progress:\n\n"
        response += f"Learned Words: {', '.join(learned_words) if learned_words else 'None'}\n"
        response += f"Words in Progress: {', '.join(learning_words) if learning_words else 'None'}\n"

        dispatcher.utter_message(response)

        return []

class ActionShowStruggles(Action):
    def name(self) -> Text:
        return "action_show_struggles"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain: Dict[Text, Any]) -> List[Dict]:
        user_id = "user123"

        struggle_data = user_struggles_collection.find_one({"user_id": user_id}) or {"words": [], "grammar": []}
        struggled_words = [word["word"] for word in struggle_data.get("words",[])]
        grammar_issues = [g["grammar_type"] for g in struggle_data.get("grammar",[])]


        response = "Your Struggles:\n\n"

        response += f"Struggled Words: {', '.join(struggled_words) if struggled_words else 'None'}\n"
        response += f"Grammar Issues: {', '.join(grammar_issues) if grammar_issues else 'None'}\n"

        dispatcher.utter_message(response)

        return []

class ActionFetchKnowledgeBase(Action):
    def name(self) -> Text:
        return "action_fetch_knowledge_base"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        entities = tracker.latest_message.get("entities", [])

        if not entities:
            dispatcher.utter_message(text="Мен ешқандай нақты атауды анықтай алмадым.")
            return []

        responses = []

        for entity in entities:
            entity_value = entity.get("value")
            entity_type = entity.get("entity")

            result = knowledge_collection.find_one({"name": entity_value, "entity": entity_type})

            if result:
                response_text = f"{entity_value}: {result.get('description', 'Бұл туралы қосымша ақпарат жоқ.')}\n"
                if "source" in result:
                    response_text += f"Көбірек оқу үшін: {result['source']}"
            else:
                response_text = f"Мен '{entity_value}' туралы білім базасында ақпарат таппадым, бірақ ол {entity_type} ретінде анықталды."

            responses.append(response_text)

        dispatcher.utter_message(text="\n\n".join(responses))
        return []

class ActionDefineWord(Action):
    def name(self):
        return "action_define_word"

    def run(self, dispatcher, tracker, domain):
        user_language = tracker.get_slot("language")
        word_to_define = tracker.get_slot("word_to_define")

        if not word_to_define:
            dispatcher.utter_message(text=self.get_message("no_word", user_language))
            return []

        word_entry = dictionary_collection.find_one({"word": word_to_define})
        if not word_entry:
            dispatcher.utter_message(text=self.get_message("word_not_found", user_language))
            return []

        definition_key = f"definition_{user_language}"
        definition = word_entry.get(definition_key, self.get_message("no_definition", user_language))

        examples = "\n".join(word_entry.get(f"example_sentences", []))

        response = self.get_message("definition_response", user_language).format(
            word=word_to_define, definition=definition, examples=examples
        )
        dispatcher.utter_message(text=response)

        return []

    def get_message(self, key, lang):
        messages = {
            "no_word": {
                "en": "Please specify a word to define.",
                "ru": "Пожалуйста, укажите слово для определения.",
                "kk": "Анықтамасын беру үшін сөзді көрсетіңіз."
            },
            "word_not_found": {
                "en": "I couldn't find the definition for that word.",
                "ru": "Я не смог найти определение этого слова.",
                "kk": "Бұл сөздің анықтамасын таба алмадым."
            },
            "no_definition": {
                "en": "Definition not available in this language.",
                "ru": "Определение на этом языке недоступно.",
                "kk": "Бұл тілде анықтама жоқ."
            },
            "definition_response": {
                "en": "Definition of '{word}': {definition}\nExamples:\n{examples}",
                "ru": "Определение слова '{word}': {definition}\nПримеры:\n{examples}",
                "kk": "'{word}' сөзінің анықтамасы: {definition}\nМысалдар:\n{examples}"
            }
        }

        return messages[key].get(lang, messages[key]["en"])


class ActionGenerateGrammarQuiz(Action):
    def name(self):
        return "action_generate_grammar_quiz"

    def run(self, dispatcher, tracker, domain):
        user_id = "user123"

        user_data = users_collection.find_one({"_id": user_id})
        level = user_data["language_level"] if user_data else "A1"

        active_quiz = quiz_results_collection.find_one(
            {"user_id": user_id, "quiz_type": "grammar", "is_active": True}
        )

        if active_quiz and active_quiz["attempt_count"] < 6:
            quiz_id = active_quiz["quiz_id"]
            attempt_count = active_quiz["attempt_count"]
        else:
            quiz_id = str(int(time.time()))
            attempt_count = 0
            quiz_results_collection.insert_one({
                "user_id": user_id,
                "quiz_id": quiz_id,
                "quiz_type": "grammar",
                "attempt_count": 0,
                "is_active": True,
                "quiz_attempts": []
            })

        quiz = grammar_quizzes_collection.find_one({"level": level})
        if not quiz:
            dispatcher.utter_message(text="No grammar quiz available for your level.")
            return []

        questions = quiz["questions"]
        question = random.choice(questions)

        quiz_results_collection.update_one(
            {"user_id": user_id, "quiz_id": quiz_id},
            {"$set": {"attempt_count": attempt_count + 1}}
        )

        dispatcher.utter_message(
            text=f"Fill in the blank: {question['sentence']}\nOptions: {', '.join(question['options'])}"
        )
        return [SlotSet("current_quiz", {
            "type": "grammar",
            "grammar_type": quiz["grammar_type"],  # Include grammar type
            "question": question,
            "correct_answer": question["correct_answer"],
            "quiz_id": quiz_id
        })]


class ActionGenerateVocabQuiz(Action):
    def name(self):
        return "action_generate_vocab_quiz"

    def run(self, dispatcher, tracker, domain):
        user_id = "user123"

        user_data = users_collection.find_one({"_id": user_id})
        level = user_data["language_level"] if user_data else "A1"

        active_quiz = quiz_results_collection.find_one(
            {"user_id": user_id, "quiz_type": "vocab", "is_active": True}
        )

        if active_quiz and active_quiz["attempt_count"] < 6:
            quiz_id = active_quiz["quiz_id"]
            attempt_count = active_quiz["attempt_count"]
        else:
            quiz_id = str(int(time.time()))
            attempt_count = 0
            quiz_results_collection.insert_one({
                "user_id": user_id,
                "quiz_id": quiz_id,
                "quiz_type": "vocab",
                "attempt_count": 0,
                "is_active": True,
                "quiz_attempts": []
            })

        quiz = vocab_quizzes_collection.find_one({"level": level})
        if not quiz:
            dispatcher.utter_message(text="No vocabulary quiz available for your level.")
            return []

        questions = quiz["questions"]
        question = random.choice(questions)

        quiz_results_collection.update_one(
            {"user_id": user_id, "quiz_id": quiz_id},
            {"$set": {"attempt_count": attempt_count + 1}}
        )

        dispatcher.utter_message(
            text=f"Translate this word: {question['word']}\nOptions: {', '.join(question['options'])}"
        )
        return [
            SlotSet("current_quiz", {
                "type": "vocab",
                "vocab_type": quiz["vocab_type"],  # Include vocab type
                "question": question,
                "correct_answer": question["correct_answer"],
                "quiz_id": quiz_id
            })
        ]

class ActionProcessQuizResponse(Action):
    def name(self):
        return "action_process_quiz_response"

    def run(self, dispatcher, tracker, domain):
        user_id = "user123"
        user_answer = tracker.get_slot("word_answer")

        if not user_answer:
            dispatcher.utter_message(text=self.get_message("no_answer"))
            return []

        quiz_data = tracker.get_slot("current_quiz")

        if not quiz_data:
            dispatcher.utter_message(text=self.get_message("no_quiz"))
            return []

        quiz_id = quiz_data["quiz_id"]
        correct_answer = quiz_data["correct_answer"]
        quiz_type = quiz_data["type"]

        is_correct = user_answer.lower() == correct_answer.lower()

        quiz_results_collection.update_one(
            {"user_id": user_id, "quiz_id": quiz_id},
            {"$push": {"quiz_attempts": {
                "question": quiz_data["question"],
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct
            }}}
        )

        quiz_session = quiz_results_collection.find_one({"user_id": user_id, "quiz_id": quiz_id})
        attempt_count = len(quiz_session["quiz_attempts"])

        if is_correct:
            dispatcher.utter_message(text=self.get_message("correct"))

            if quiz_type == "vocab":
                mastering_collection.update_one(
                    {"user_id": user_id},
                    {"$push": {"words": {"word": quiz_data["question"]["word"], "attempts": 1, "status": "learning"}}},
                    upsert=True
                )

        else:
            dispatcher.utter_message(text=self.get_message("incorrect").format(correct_answer=correct_answer))

            if quiz_type == "vocab":
                user_struggles_collection.update_one(
                    {"user_id": user_id},
                    {"$push": {"words": {"word": quiz_data["question"]["word"], "failed_attempts": 1}}},
                    upsert=True
                )

            elif quiz_type == "grammar":
                grammar_type = quiz_data["grammar_type"]
                user_struggles_collection.update_one(
                    {"user_id": user_id},
                    {"$push": {"grammar": {"grammar_type": grammar_type, "failed_attempts": 1}}},
                    upsert=True
                )

        if attempt_count >= 4:
            quiz_results_collection.update_one(
                {"user_id": user_id, "quiz_id": quiz_id},
                {"$set": {"is_active": False}}
            )
            dispatcher.utter_message(text=self.get_message("quiz_finished"))
        else:
            dispatcher.utter_message(text=self.get_message("next_question"))

        return []

    def get_message(self, key):
        messages = {
            "no_answer": {
                "en": "You didn't answer properly, sorry for that.",
                "ru": "Вы ответили неправильно, извините за это.",
                "kk": "Сіз дұрыс жауап бермедіңіз, кешіріңіз."
            },
            "no_quiz": {
                "en": "No active quiz question.",
                "ru": "Нет активного вопроса викторины.",
                "kk": "Белсенді викторина сұрағы жоқ."
            },
            "correct": {
                "en": "Correct! Well done.",
                "ru": "Правильно! Молодец.",
                "kk": "Дұрыс! Жарайсың."
            },
            "incorrect": {
                "en": "Incorrect. The correct answer is: {correct_answer}",
                "ru": "Неправильно. Правильный ответ: {correct_answer}",
                "kk": "Қате. Дұрыс жауап: {correct_answer}"
            },
            "quiz_finished": {
                "en": "Quiz finished! Detecting your level... Do you want to see results?",
                "ru": "Викторина завершена! Определяю ваш уровень... Хотите увидеть результаты?",
                "kk": "Викторина аяқталды! Деңгейіңізді анықтаймын... Нәтижелерді көргіңіз келе ме?"
            },
            "next_question": {
                "en": "Okay, next question. Are you ready? Say 'next vocab' or 'next grammar'.",
                "ru": "Хорошо, следующий вопрос. Вы готовы? Скажите 'следующее слово' или 'следующая грамматика'.",
                "kk": "Жақсы, келесі сұрақ. Дайынсыз ба? 'Келесі сөз' немесе 'Келесі грамматика' деп айтыңыз."
            }
        }

        user_language = "en"  # Change this based on the user's language preference
        return messages[key].get(user_language, messages[key]["en"])


class ActionDetectUserLevel(Action):
    def name(self):
        return "action_detect_user_level"

    def run(self, dispatcher, tracker, domain):
        user_id = "user123"
        language = tracker.get_slot("language")

        quiz_results = list(quiz_results_collection.find({"user_id": user_id}))

        if not quiz_results:
            message = (
                "No quiz data found to assess your level." if language == "en" else
                "Нет данных о викторинах для определения вашего уровня." if language == "ru" else
                "Сіздің деңгейіңізді бағалау үшін викторина деректері табылмады."
            )
            dispatcher.utter_message(text=message)
            return []

        total_attempts = 0
        correct_answers = 0

        for quiz in quiz_results:
            total_attempts += len(quiz["quiz_attempts"])
            correct_answers += sum(1 for attempt in quiz["quiz_attempts"] if attempt["is_correct"])

        accuracy = (correct_answers / total_attempts) * 100 if total_attempts > 0 else 0

        if accuracy >= 90:
            level = "C1"
        elif accuracy >= 75:
            level = "B2"
        elif accuracy >= 60:
            level = "B1"
        elif accuracy >= 40:
            level = "A2"
        else:
            level = "A1"

        users_collection.update_one({"_id": user_id}, {"$set": {"language_level": level}}, upsert=True)

        response = (
            f"Your detected level is now: {level}" if language == "en" else
            f"Ваш определенный уровень: {level}" if language == "ru" else
            f"Сіздің анықталған деңгейіңіз: {level}"
        )

        dispatcher.utter_message(text=response)
        return []

class ActionTilmashTranslate(Action):
    def name(self) -> Text:
        return "action_tilmash_translate"

    @staticmethod
    def tilmash_translate(text: str, source_lang: str) -> str:
        try:
            HF_TOKEN = os.getenv("HF_TOKEN")

            if not HF_TOKEN:
                return "Error: Hugging Face token is missing."

            login(HF_TOKEN)

            model = AutoModelForSeq2SeqLM.from_pretrained("issai/tilmash")
            tokenizer = AutoTokenizer.from_pretrained("issai/tilmash", src_lang=source_lang)

            inputs = tokenizer(text, return_tensors='pt')

            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("kaz_Cyrl"),
                max_length=1000
            )
            translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translated_text = translated_text[0]
            return translated_text

        except Exception as e:
            return f"Translation error: {str(e)}"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]) -> List[Dict]:
        text_to_translate = tracker.get_slot("sentence_to_translate")
        source_lang = tracker.get_slot("language")

        if not text_to_translate:
            message = (
                "No text provided for translation."
                if source_lang != "ru" else
                "Текст для перевода не предоставлен."
            )
            dispatcher.utter_message(text=message)
            return []

        source_lang = source_lang.lower() if source_lang else "eng_Latn"

        lang_mapping = {
            "en": "eng_Latn",
            "rus": "rus_Cyrl"
        }

        source_lang_code = lang_mapping.get(source_lang, "eng_Latn")  # Default to English

        translation = self.tilmash_translate(text_to_translate, source_lang_code)

        response = (
            f"Translation (Kazakh): {translation}"
            if source_lang != "ru" else
            f"Перевод (на казахский): {translation}"
        )

        dispatcher.utter_message(text=response)
        return [SlotSet("translation_result", translation)]


class DetectLanguage(Action):
    def name(self):
        return "action_detect_language"

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message.get("text")
        if not user_message:
            return []

        try:
            detected_lang = detect(user_message)
            if detected_lang in ["en"]:
                detected_lang = "en"
            elif detected_lang in ["ru"]:
                detected_lang = "ru"
            else:
                detected_lang = "kz"
        except:
            detected_lang = "en"

        current_language = tracker.get_slot("language")
        if current_language != detected_lang:
            return [SlotSet("language", detected_lang)]

        return []

class ActionTestMongo(Action):
    def name(self):
        return "action_test_mongo"

    def run(self, dispatcher, tracker, domain):
        user_data = users_collection.find_one({"user_id": "12345"})  # Fetch user

        if user_data:
            response = f"Hello {user_data['name']}! You're using {user_data['language']}. Words learned: {', '.join(user_data['words_learned'])}."
        else:
            response = "No user data found."

        dispatcher.utter_message(text=response)
        return []

class ActionGenerateCaseExamples(Action):
    def name(self) -> Text:
        return "action_generate_case_examples_for_noun"

    @staticmethod
    def determine_vowel_type(word: str) -> str:

        thick_vowels = "аоуы"
        thin_vowels = "еөүі"

        for char in reversed(word):
            if char in thick_vowels:
                return "thick"
            if char in thin_vowels:
                return "thin"
        return "unknown"

    @staticmethod
    def generate_accusative(noun: str) -> str:
        last_letter = noun[-1]
        vowel_type = ActionGenerateCaseExamples.determine_vowel_type(noun)

        if last_letter in "нмрлжзийуюң":
            return noun + ("ды" if vowel_type == "thick" else "ді")
        elif last_letter in "аеёоыэяөіү":
            return noun + ("ны" if vowel_type == "thick" else "ні")
        else:
            return noun + ("ты" if vowel_type == "thick" else "ті")

    @staticmethod
    def generate_dative(noun: str) -> str:
        last_letter = noun[-1]
        vowel_type = ActionGenerateCaseExamples.determine_vowel_type(noun)

        if last_letter in "нмрлжз" or last_letter in "аеёиоуыэюяөіү":
            return noun + ("ға" if vowel_type == "thick" else "ге")
        else:
            return noun + ("қа" if vowel_type == "thick" else "ке")

    @staticmethod
    def generate_genitive(noun: str) -> str:
        last_letter = noun[-1]
        vowel_type = ActionGenerateCaseExamples.determine_vowel_type(noun)

        if last_letter in "нмң" or last_letter in "аеёоыэяөі":
            return noun + ("ның" if vowel_type == "thick" else "нің")
        elif last_letter in "рлжзийую":
            return noun + ("дың" if vowel_type == "thick" else "дің")
        else:
            return noun + ("тың" if vowel_type == "thick" else "тің")

    @staticmethod
    def generate_locative(noun: str) -> str:
        last_letter = noun[-1]
        vowel_type = ActionGenerateCaseExamples.determine_vowel_type(noun)

        if last_letter in "нмрлжз" or last_letter in "аеёиоуыэюяөіү":
            return noun + ("да" if vowel_type == "thick" else "де")
        else:
            return noun + ("та" if vowel_type == "thick" else "те")

    @staticmethod
    def generate_ablative(noun: str) -> str:
        last_letter = noun[-1]
        vowel_type = ActionGenerateCaseExamples.determine_vowel_type(noun)

        if last_letter in "рлжз" or last_letter in "аеёиоуыэюяөіү":
            return noun + ("дан" if vowel_type == "thick" else "ден")
        elif last_letter in "нм":
            return noun + ("нан" if vowel_type == "thick" else "нен")
        else:
            return noun + ("тан" if vowel_type == "thick" else "тен")

    @staticmethod
    def generate_instrumental(noun: str) -> str:
        last_letter = noun[-1]
        vowel_type = ActionGenerateCaseExamples.determine_vowel_type(noun)

        if last_letter in "мн":
            return noun + "мен"
        elif last_letter in "рлжз" or last_letter in "аеёиоуыэюяөіү":
            return noun + "бен"
        else:
            return noun + "пен"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[Dict]:
        noun = tracker.get_slot("noun_to_decline")
        language = tracker.get_slot("language")

        if not noun:
            if language == "ru":
                message = "Пожалуйста, укажите существительное для склонения."
            elif language == "kk":
                message = "Септеу үшін зат есім енгізіңіз."
            else:
                message = "Please provide a noun to generate case examples."
            dispatcher.utter_message(text=message)
            return []

        cases = {
            "Accusative": self.generate_accusative(noun),
            "Dative": self.generate_dative(noun),
            "Genitive": self.generate_genitive(noun),
            "Locative": self.generate_locative(noun),
            "Ablative": self.generate_ablative(noun),
            "Instrumental": self.generate_instrumental(noun),
        }

        if language == "ru":
            case_names = {
                "Accusative": "Табыс септік (Винительный падеж)",
                "Dative": "Барыс септік (Дательный падеж)",
                "Genitive": "Ілік септік (Родительный падеж)",
                "Locative": "Жатыс септік (Предложный падеж)",
                "Ablative": "Шығыс септік (Исходный падеж)",
                "Instrumental": "Көмектес септік (Творительный падеж)"
            }
            response = "Примеры склонения для '{}':\n\n".format(noun) + "\n".join(
                f"{case_names[case]}: {cases[case]}" for case in cases)
        elif language == "kk":
            case_names = {
                "Accusative": "Табыс септік",
                "Dative": "Барыс септік",
                "Genitive": "Ілік септік",
                "Locative": "Жатыс септік",
                "Ablative": "Шығыс септік",
                "Instrumental": "Көмектес септік"
            }
            response = "'{}' сөзінің септелу мысалдары:\n\n".format(noun) + "\n".join(
                f"{case_names[case]}: {cases[case]}" for case in cases)
        else:
            response = "Case Examples for '{}':\n\n".format(noun) + "\n".join(f"{case}: {cases[case]}" for case in cases)

        dispatcher.utter_message(text=response)
        return [SlotSet("case_examples", cases)]

class ActionExplainAdjectives(Action):
    def name(self) -> Text:
        return "action_explain_adjectives"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[Dict]:
        language = tracker.get_slot("language")

        if language == "ru":
            explanation = (
                "В казахском языке прилагательные (сын есім) описывают свойства предметов и бывают нескольких видов:\n\n"
                "1. Качественные (Сапалық сын есім) – обозначают признаки предмета, могут изменяться по степеням (*үлкен – больше, ең үлкен – самый большой*).\n"
                "2. Относительные (Қатыстық сын есім) – образованы от других слов и обозначают материал, время и т. д. (*металлдан жасалған – металлический, қыстық – зимний*).\n\n"
                "Прилагательные также могут изменяться по степеням сравнения:\n\n"
                "Салыстырмалы (Сравнительная степень) – образуется с помощью суффиксов *-рақ, -рек, -лау, -леу* (*жақсы – жақсырақ*).\n"
                "Күшейтпелі (Усилительная степень) – усиливает признак, добавляя слова типа *өте, аса, тым* (*өте әдемі* – очень красивый).\n\n"
                "Хотите разобрать конкретное прилагательное?"
            )
        elif language == "kk":
            explanation = (
                "Қазақ тілінде сын есімдер заттың қасиетін білдіреді және бірнеше түрге бөлінеді:\n\n"
                "1. Сапалық сын есім – заттың өзіндік белгісін білдіреді, шырай түрлерінде түрленеді (*үлкен – үлкенірек – ең үлкен*).\n"
                "2. Қатыстық сын есім – басқа сөздерден жасалып, заттың қатыстылығын білдіреді (*металлдан жасалған – металдық, қыс – қыстық*).\n\n"
                "Сын есімнің шырай түрлері:\n\n"
                "Салыстырмалы шырай – *-рақ, -рек, -лау, -леу* жұрнақтары арқылы жасалады (*жақсы – жақсырақ*).\n"
                "Күшейтпелі шырай – белгіні күшейтеді, *өте, аса, тым* сөздерімен беріледі (*өте әдемі* – өте сұлу*).\n\n"
                "Белгілі бір сын есімді талдауды қалайсыз ба?"
            )
        else:
            explanation = (
                "In Kazakh, adjectives (*сын есім*) describe qualities of objects and have different types:\n\n"
                "1. Qualitative adjectives (Сапалық сын есім) – indicate inherent qualities and can change by degree (*үлкен – bigger – biggest*).\n"
                "2. Relative adjectives (Қатыстық сын есім) – derived from other words and indicate material, time, etc. (*metallic – winter-related*).\n\n"
                "Adjectives also have degrees of comparison:\n\n"
                "Comparative degree (Салыстырмалы шырай) – formed with suffixes *-рақ, -рек, -лау, -леу* (*better – much better*).\n"
                "Intensified degree (Күшейтпелі шырай) – intensifies meaning using words like *very, extremely* (*very beautiful*).\n\n"
                "Would you like to analyze a specific adjective?"
            )

        dispatcher.utter_message(text=explanation)
        return []

class ActionExplainNouns(Action):
    def name(self) -> Text:
        return "action_explain_nouns"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[Dict]:
        language = tracker.get_slot("language")

        if language == "ru":
            explanation = (
                "В казахском языке существительные изменяются по падежам, чтобы показать свою роль в предложении. "
                "Существует 7 падежей:\n\n"
                "1. Именительный (Атау септік): Основная форма (*кітап* - книга)\n"
                "2. Родительный (Ілік септік): Принадлежность (*кітаптың* - книги, *кітаптың авторы* - автор книги)\n"
                "3. Дательный (Барыс септік): Направление (*кітапқа* - к книге)\n"
                "4. Винительный (Табыс септік): Прямое дополнение (*кітапты* - книгу)\n"
                "5. Местный (Жатыс септік): Местоположение (*кітапта* - в книге)\n"
                "6. Исходный (Шығыс септік): Источник (*кітаптан* - из книги)\n"
                "7. Творительный (Көмектес септік): Орудие (*кітаппен* - с книгой)\n\n"
                "Хотите разобрать конкретное существительное?"
            )
        elif language == "kk":
            explanation = (
                "Қазақ тілінде зат есімдер сөйлемдегі рөлін көрсету үшін септеледі. "
                "Барлығы 7 септік бар:\n\n"
                "1. Атау септік (Nominative): Негізгі түрі (*кітап* - кітап)\n"
                "2. Ілік септік (Genitive): Тиістілік (*кітаптың* - кітаптың, *кітаптың авторы* - кітаптың авторы)\n"
                "3. Барыс септік (Dative): Бағыт (*кітапқа* - кітапқа)\n"
                "4. Табыс септік (Accusative): Тура толықтауыш (*кітапты* - кітапты)\n"
                "5. Жатыс септік (Locative): Орын (*кітапта* - кітапта)\n"
                "6. Шығыс септік (Ablative): Бастау нүктесі (*кітаптан* - кітаптан)\n"
                "7. Көмектес септік (Instrumental): Құрал (*кітаппен* - кітаппен)\n\n"
                "Белгілі бір зат есімді талдауды қалайсыз ба?"
            )
        else:
            explanation = (
                "In Kazakh, nouns change based on cases to indicate their role in a sentence. "
                "There are 7 cases:\n\n"
                "1. Атау септік (Nominative): Basic form (e.g., *кітап* - book)\n"
                "2. Ілік септік (Genitive): Shows possession (*кітаптың* - of the book)\n"
                "3. Барыс септік (Dative): Direction (*кітапқа* - to the book)\n"
                "4. Табыс септік (Accusative): Direct object (*кітапты* - the book)\n"
                "5. Жатыс септік (Locative): Location (*кітапта* - in the book)\n"
                "6. Шығыс септік (Ablative): Source (*кітаптан* - from the book)\n"
                "7. Көмектес септік (Instrumental): Tool (*кітаппен* - with the book)\n\n"
                "Would you like to analyze a specific noun?"
            )

        dispatcher.utter_message(text=explanation)
        return []

class ActionAnalyzeNoun(Action):
    def name(self) -> Text:
        return "action_analyze_noun"

    @staticmethod
    def apertium_pos_tagging(noun: str) -> str:
        try:
            command = f"echo '{noun}' | apertium kaz-tagger"
            result = subprocess.run(
                ["wsl", "bash", "-c", command],  # Runs inside WSL
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[Dict]:
        noun = tracker.get_slot("noun_to_analyze")
        language = tracker.get_slot("language")

        if not noun:
            message = "Please provide a noun to analyze." if language != "ru" else "Пожалуйста, укажите существительное для анализа."
            dispatcher.utter_message(text=message)
            return []

        pos_tags = self.apertium_pos_tagging(noun)

        case_mappings = {
            "<nom>": ("Nominative (Атау септік)", "Именительный (Атау септік)"),
            "<gen>": ("Genitive (Ілік септік)", "Родительный (Ілік септік)"),
            "<dat>": ("Dative (Барыс септік)", "Дательный (Барыс септік)"),
            "<acc>": ("Accusative (Табыс септік)", "Винительный (Табыс септік)"),
            "<loc>": ("Locative (Жатыс септік)", "Местный (Жатыс септік)"),
            "<abl>": ("Ablative (Шығыс септік)", "Исходный (Шығыс септік)"),
            "<ins>": ("Instrumental (Көмектес септік)", "Творительный (Көмектес септік)"),
        }

        detected_cases = [case_mappings[tag][1] if language == "ru" else case_mappings[tag][0] for tag in case_mappings if tag in pos_tags]
        detected_cases_text = ", ".join(detected_cases) if detected_cases else ("No noun cases detected." if language != "ru" else "Падежи не обнаружены.")

        if language == "ru":
            response = f"Анализ **{noun}**:\nОбнаруженные падежи: {detected_cases_text}\n\nPOS-теги: {pos_tags}"
        else:
            response = f"Analysis of **{noun}**:\nDetected cases: {detected_cases_text}\n\nPOS Tags: {pos_tags}"

        dispatcher.utter_message(text=response)
        return [SlotSet("noun_analysis_result", pos_tags)]

import subprocess
import re
from typing import Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

class ActionAnalyzeNouns(Action):
    def name(self) -> Text:
        return "action_analyze_nouns"

    @staticmethod
    def apertium_pos_tagging(text: str) -> str:
        try:
            command = f"echo '{text}' | apertium kaz-tagger"
            result = subprocess.run(
                ["wsl", "bash", "-c", command],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def extract_nouns(self, pos_tags: str) -> List[str]:
        noun_pattern = r"\^([\w-]+)\/[\w-]+<n>.*?\$"
        return re.findall(noun_pattern, pos_tags)

    def detect_cases(self, pos_tags: str, language: str) -> List[str]:
        case_mappings = {
            "<nom>": ("Nominative (Атау септік)", "Именительный (Атау септік)"),
            "<gen>": ("Genitive (Ілік септік)", "Родительный (Ілік септік)"),
            "<dat>": ("Dative (Барыс септік)", "Дательный (Барыс септік)"),
            "<acc>": ("Accusative (Табыс септік)", "Винительный (Табыс септік)"),
            "<loc>": ("Locative (Жатыс септік)", "Местный (Жатыс септік)"),
            "<abl>": ("Ablative (Шығыс септік)", "Исходный (Шығыс септік)"),
            "<ins>": ("Instrumental (Көмектес септік)", "Творительный (Көмектес септік)"),
        }
        return [case_mappings[tag][1] if language == "ru" else case_mappings[tag][0] for tag in case_mappings if tag in pos_tags]

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[Dict]:
        input_text = tracker.get_slot("text_to_analyze")
        language = tracker.get_slot("language")

        if not input_text:
            message = (
                "Please provide a noun or a sentence to analyze."
                if language != "ru" else
                "Пожалуйста, укажите существительное или предложение для анализа."
            )
            dispatcher.utter_message(text=message)
            return []

        pos_tags = self.apertium_pos_tagging(input_text)
        nouns = self.extract_nouns(pos_tags)

        if not nouns:
            message = (
                "No nouns detected in the given input."
                if language != "ru" else
                "В данном тексте существительные не обнаружены."
            )
            dispatcher.utter_message(text=message)
            return []

        analysis_results = {}
        for noun in nouns:
            cases = self.detect_cases(pos_tags, language)
            analysis_results[noun] = cases if cases else (
                ["No specific cases detected"] if language != "ru" else ["Падежи не обнаружены"]
            )

        response = "**Noun Analysis:**\n\n" if language != "ru" else "**Анализ существительных:**\n\n"
        for noun, cases in analysis_results.items():
            response += f"**{noun}** → {', '.join(cases)}\n"

        dispatcher.utter_message(text=response)
        return [SlotSet("noun_analysis_result", analysis_results)]

class ActionApertiumPOSTagging(Action):
    def name(self) -> Text:
        return "action_apertium_pos_tagging"

    @staticmethod
    def apertium_pos_tagging(text: str) -> str:
        try:
            command = f"echo '{text}' | apertium kaz-tagger"
            result = subprocess.run(
                ["wsl", "bash", "-c", command],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict) -> List[Dict]:
        text_to_tag = tracker.get_slot("sentence_to_tag")
        if not text_to_tag:
            dispatcher.utter_message(text="No text provided for POS tagging.")
            return []

        pos_tags = self.apertium_pos_tagging(text_to_tag)
        dispatcher.utter_message(text=f"POS Tags: {pos_tags}")
        return [SlotSet("pos_tags_result", pos_tags)]




