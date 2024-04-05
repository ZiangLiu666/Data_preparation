import random
import numpy as np
import pandas as pd
from tasks.basis import TabularCorruption


class TextNoise(TabularCorruption):
    def __init__(self, column, fraction, sampling='CAR'):
        super().__init__(column, fraction, sampling)

    def transform(self, data):
        df = data.copy(deep=True)

        if self.fraction > 0:
            rows = self.sample_rows(data)
            for row in rows:
                original_text = df.loc[row, self.column]
                if pd.isnull(original_text):
                    continue
                # Adding word-level noise along with character-level
                noised_text = self.add_word_level_noise(original_text) if random.random() < 0.5 else self.add_typo(
                    original_text)
                df.loc[row, self.column] = noised_text

        return df

    def add_typo(self, text):
        # 随机选择一个位置插入、删除或替换字符来模拟打字错误
        if len(text) < 1:
            return text

        operations = ['insert', 'delete', 'replace']
        operation = random.choice(operations)

        position = random.randint(0, len(text) - 1)
        if operation == 'insert':
            char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
            text = text[:position] + char_to_insert + text[position:]
        elif operation == 'delete' and len(text) > 1:
            text = text[:position] + text[position + 1:]
        elif operation == 'replace':
            char_to_replace = random.choice('abcdefghijklmnopqrstuvwxyz')
            text = text[:position] + char_to_replace + text[position + 1:]

        return text

    def add_word_level_noise(self, text):
        words = text.split()
        # if the text has less than 2 words,retrun the original text
        if len(words) < 2:
            return text

        operations = ['swap', 'delete', 'duplicate', 'reverse']
        operation = random.choice(operations)

        if operation == 'swap':
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        elif operation == 'delete':
            del words[random.randint(0, len(words) - 1)]
        elif operation == 'duplicate':
            idx = random.randint(0, len(words) - 1)
            words.insert(idx, words[idx])
        elif operation == 'reverse':
            idx1, idx2 = sorted(random.sample(range(len(words)), 2))
            words = words[:idx1] + words[idx1:idx2 + 1][::-1] + words[idx2 + 1:]

        return ' '.join(words)