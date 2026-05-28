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

    # def respond(self, conversation_history, config):
    #     """Generuje odpowiedź na podstawie historii rozmowy.

    #     Returns:
    #         (str, int) — odpowiedź i liczba tokenów
    #     """
    #     debate_so_far = "\n\n".join(conversation_history)
    #     messages = [
    #         {"role": "system", "content": self.system_prompt},
    #         {"role": "user", "content": debate_so_far + "\n\nTwoja odpowiedź:"},
    #     ]
    #     return _generate(self.model, self.tokenizer, messages, config)
    def respond(self, conversation_history, config):
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in conversation_history:
            # zakładamy format: "Agent: tekst"
            if ": " in msg:
                agent, text = msg.split(": ", 1)
                role = "assistant" if agent == self.name else "user"
                messages.append({"role": role, "content": f"{agent}: {text}"})
            else:
                messages.append({"role": "user", "content": msg})

        messages.append({"role": "user", "content": "Teraz kolej na twoją odpowiedź w tej rozmowie. "})

        return _generate(self.model, self.tokenizer, messages, config)
        