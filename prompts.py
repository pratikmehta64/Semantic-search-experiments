class Prompt(object):
    def __init__(self, prompt: str):
        self.prompt = prompt

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return f"Prompt(prompt={self.prompt})"

prompt_1 = Prompt(
                    """ 
                    You are an expert in search engine evaluation. 
                    Find the most relevant search results for the following search query. 
                    The hard criteria is your topmost priority and has to always be satisfied. 
                    """
                )

