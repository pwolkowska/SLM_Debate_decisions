"""
agents.py — Agent do multi-agent debate.

Każdy Agent ma swój system_prompt, ale współdzieli model z innymi agentami.
_generate zwraca (tekst, liczba_tokenów) żeby metryki były liczone na bieżąco.
"""

import torch


def _generate(model, tokenizer, messages, config):
    """Generuje odpowiedź z chat template.

    Returns:
        (str, int) — wygenerowany tekst i liczba nowych tokenów
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", 256),
            max_length=None,
            temperature=config.get("temperature", 0.7),
            do_sample=config.get("do_sample", True),
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, len(new_tokens)


class Agent:
    """Jeden uczestnik debaty."""

    def __init__(self, name, system_prompt, model, tokenizer):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tokenizer = tokenizer

    def respond(self, conversation_history, config):
        topic_line = conversation_history[0]
        turns = conversation_history[1:]

        # Zbuduj listę tur: (nazwa_agenta, tekst)
        parsed = []
        for turn in turns:
            if ': ' in turn:
                speaker, content = turn.split(': ', 1)
                parsed.append((speaker.strip(), content.strip()))

        # Zbuduj messages — historia jako naprzemienne user/assistant
        # Własne tury → assistant, obce → user
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"{topic_line}\n\n"
                "Wypowiadaj się WYŁĄCZNIE we własnym imieniu, w pierwszej osobie. "
                "NIE pisz 'Osoba 1:', 'Osoba 2:' ani żadnych etykiet. "
                "NIE streszczaj historii. Zacznij od razu od swojego argumentu."
            )},
        ]

        # Wstrzyknij historię jako dialog
        for speaker, content in parsed:
            if speaker == self.name:
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({
                    "role": "user",
                    "content": f"{speaker} powiedział: {content}\n\nOdpowiedz na ten argument."
                })

        # Upewnij się że ostatnia wiadomość to user
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({
                "role": "user",
                "content": "Twoja kolej. Przedstaw swój argument w 3-4 zdaniach."
            })

        return _generate(self.model, self.tokenizer, messages, config)
        # def respond(self, conversation_history, config):
        """Generuje odpowiedź na podstawie historii rozmowy.

        Returns:
            (str, int) — odpowiedź i liczba tokenów
        """
        # debate_so_far = "\n\n".join(conversation_history)
        # messages = [
        #     {"role": "system", "content": self.system_prompt},
        #     {"role": "user", "content": debate_so_far + "\n\nTwoja odpowiedź:"},
        # ]
        # return _generate(self.model, self.tokenizer, messages, config)

    # def respond(self, conversation_history, config):
    #     """Generuje odpowiedź na podstawie historii rozmowy.

    #     Returns:
    #         (str, int) — odpowiedź i liczba tokenów
    #     """

    #     # pełna historia, ale jako dialog z rolami
    #     formatted_history = "\n".join(conversation_history)

    #     messages = [
    #         {
    #             "role": "system",
    #             "content": (
    #                 self.system_prompt
    #             )
    #         },
    #         {
    #             "role": "user",
    #             "content": (
    #                 "Jesteś "+ self.name +
    #                 "\nPoniżej znajduje się pełna dotychczasowa rozmowa:\n\n"
    #                 f"{formatted_history}\n\n"
    #                 "Napisz kolejną wypowiedź w tej rozmowie."
    #             )
    #         },
    #     ]

    #     return _generate(self.model, self.tokenizer, messages, config)
#     def respond(self, conversation_history, config):
#         """Generuje odpowiedź na podstawie historii rozmowy.
#
#         Returns:
#             (str, int) — odpowiedź i liczba tokenów
#         """
#
#         formatted_history = "\n".join(conversation_history)
#
#         messages = [
#             {
#                 "role": "system",
#                 "content": (
#                     self.system_prompt
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": (
#                     "Odpowiadasz teraz jako  "+ self.name +". "
#                     "Poniżej znajduje się dotychczasowa rozmowa:\n\n"
#                     f"{formatted_history}\n\n"
#                     "Napisz kolejną wypowiedź w tej rozmowie."
#                 )
# }
#         ]
#
#         return _generate(self.model, self.tokenizer, messages, config)