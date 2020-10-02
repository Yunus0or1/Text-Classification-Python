import random

# wg means word generation
# This means there might be mistakes while typing the key of this
# map and results are corresponding values.
keyBoardCharacterMapping = {
    'a': {'s', 'z', 'e', 'o', 'ae', 'aa', ''},
    'b': {'v', 'n', 'h', 'bb', ''},
    'c': {'v', 'x', 'f', 's', ''},
    'd': {'f', 's', 'c', ''},
    'e': {'r', 'w', 't', 'a', 'o', 'ee', 'ea', ''},
    'f': {'g', 'd', 'p', 'ff', ''},
    'g': {'h', 'f', 'j', 'z', ''},
    'h': {'g', 'j', 'hh', ''},
    'i': {'o', 'p', 'u', 'y', 'ai', 'ei', ''},
    'j': {'k', 'h', 'g', 'z', ''},
    'k': {'j', 'l', 'ke', 'ka', ''},
    'l': {'k', 'll', 'le', 'la', ''},
    'm': {'n', 'k', 'mm', ''},
    'n': {'m', 'b', 'nn', ''},
    'o': {'i', 'p', 'a', 'e', 'oi', 'oa', 'oe', ''},
    'p': {'o', 'f', 'pp', ''},
    'q': {'w', 'e', ''},
    'r': {'e', 't', 'y', ''},
    's': {'a', 'd', 'f', 'c', 'ss', ''},
    't': {'r', 'y', ''},
    'u': {'y', 'i', 'ou', 'au', 'eu', ''},
    'v': {'b', 'c', ''},
    'w': {'q', 'e', 't', ''},
    'x': {'c', 'z', ''},
    'y': {'t', 'u', 'i', 'ay', 'ey', ''},
    'z': {'x', 'i', 'g', 'j', ''},
}

upper_row_KeyBoard = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
middle_row_KeyBoard = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']
bottom_row_KeyBoard = ['z', 'x', 'c', 'v', 'b', 'n', 'm']


def filter_raw_data():
    inputFile = open("raw_data.txt", "r")
    outputFile = open("data.txt", "w")

    word_list = []
    for line in inputFile:
        word = line.split()

        if ((word[0] not in word_list) and (len(word[0]) > 3)):
            word_list.append(word[0])
        else:
            pass
            # print(word[0])

    for single_word in word_list:
        outputFile.write(single_word.lower() + '\n')

    outputFile.close()


def correctWord_to_wrongWord_with_space_and_no_character_position():
    inputFile = open("data.txt", "r")
    outputFile = open("mldata.txt", "w")

    for line in inputFile:
        correct_word = line.split()

        wrong_words_list = noise_maker(correct_word[0].lower())
        for single_wrong_word in wrong_words_list:
            temp_wrong_word = ''
            for i in range(0, len(single_wrong_word)):

                if i < (len(single_wrong_word) - 1):
                    temp_wrong_word = temp_wrong_word + single_wrong_word[i] + single_wrong_word[i] + ' '
                else:
                    temp_wrong_word = temp_wrong_word + single_wrong_word[i] + single_wrong_word[i] + ' '

            changed_single_wrong_word_with_correct_word = single_wrong_word + ',' + temp_wrong_word + ',' + \
                                                          correct_word[0] + "\n"
            outputFile.write(changed_single_wrong_word_with_correct_word)

    outputFile.close()
    inputFile.close()


def correctWord_to_wrongWord_with_space_and_character_position():
    inputFile = open("data.txt", "r")
    outputFile = open("mldata.txt", "w")

    for line in inputFile:
        correct_word = line.split()

        wrong_words_list = noise_maker(correct_word[0].lower())
        for single_wrong_word in wrong_words_list:
            temp_wrong_word = ''
            for i in range(0, len(single_wrong_word)):

                if i < (len(single_wrong_word) - 1):
                    temp_wrong_word = temp_wrong_word + single_wrong_word[i] + single_wrong_word[i] + ' ' + \
                                      single_wrong_word[i] + str(i) + ' '
                else:
                    temp_wrong_word = temp_wrong_word + single_wrong_word[i] + single_wrong_word[i] + ' ' + \
                                      single_wrong_word[i] + str(i)

            changed_single_wrong_word_with_correct_word = single_wrong_word + ',' + temp_wrong_word + ',' + \
                                                          correct_word[0].lower() + "\n"
            outputFile.write(changed_single_wrong_word_with_correct_word)

    outputFile.close()
    inputFile.close()


def correctWord_to_wrongWord_with_both_system():
    inputFile = open("data.txt", "r")
    outputFile = open("mldata.txt", "w")

    for line in inputFile:
        correct_word = line.split()

        wrong_words_list = noise_maker(correct_word[0].lower())
        for single_wrong_word in wrong_words_list:
            temp_wrong_word1 = ''
            temp_wrong_word2 = ''
            for i in range(0, len(single_wrong_word)):

                if i < (len(single_wrong_word) - 1):
                    temp_wrong_word1 = temp_wrong_word1 + single_wrong_word[i] + single_wrong_word[i] + ' ' + \
                                       single_wrong_word[i] + str(i) + ' '
                    temp_wrong_word2 = temp_wrong_word2 + single_wrong_word[i] + single_wrong_word[i] + ' '
                else:
                    temp_wrong_word1 = temp_wrong_word1 + single_wrong_word[i] + single_wrong_word[i] + ' ' + \
                                       single_wrong_word[i] + str(i)
                    temp_wrong_word2 = temp_wrong_word2 + single_wrong_word[i] + single_wrong_word[i]

            changed_single_wrong_word_with_correct_word = single_wrong_word + ',' + \
                                                          temp_wrong_word2 + ',' + \
                                                          temp_wrong_word1 + ',' + \
                                                          correct_word[0].lower() + "\n"
            outputFile.write(changed_single_wrong_word_with_correct_word)

    outputFile.close()
    inputFile.close()


def noise_maker(word):
    wrong_word_list = []

    wrong_word_list.append(swap_letter((word)))
    wrong_word_list.append(add_new_letter(word))
    wrong_word_list = wrong_word_list + replace_with_keyBoard_character(word)

    processed_word = remove_bottom_row_letter_for_upper_row_letter(word)
    if (processed_word != ''):
        wrong_word_list.append(remove_bottom_row_letter_for_upper_row_letter(word))
    processed_word = remove_upper_row_letter_for_bottom_row_letter(word)
    if (processed_word != ''):
        wrong_word_list.append(remove_upper_row_letter_for_bottom_row_letter(word))

    if (len(word) > 6):
        wrong_word_list = wrong_word_list + remove_two_letters(word)

    return wrong_word_list


def remove_two_letters(word):
    wrong_word_list = []

    rand1 = random.randrange(0, int(len(word) / 2))
    if (rand1 == 0):
        rand1 = 1
    rand2 = random.randrange(int(len(word) / 2), len(word))

    temp_wrong_word = list(word)
    temp_wrong_word[rand1] = ''
    temp_wrong_word[rand2] = ''

    wrong_word = "".join(temp_wrong_word)

    wrong_word_list.append(wrong_word)
    wrong_word_list = wrong_word_list + replace_with_keyBoard_character(wrong_word)
    return wrong_word_list


def remove_bottom_row_letter_for_upper_row_letter(word):
    temp_wrong_word = ""

    for i in range(0, len(word) - 1):

        if ((word[i] in bottom_row_KeyBoard) and (word[i - 1] in upper_row_KeyBoard)):
            temp_wrong_word = temp_wrong_word + ''
        else:
            temp_wrong_word = temp_wrong_word + word[i]

    temp_wrong_word = temp_wrong_word + word[len(word) - 1]

    if (word == temp_wrong_word):
        temp_wrong_word = ""

    return temp_wrong_word


def remove_upper_row_letter_for_bottom_row_letter(word):
    temp_wrong_word = word[0]

    for i in range(1, len(word)):

        if ((word[i] in upper_row_KeyBoard) and (word[i - 1] in bottom_row_KeyBoard)):
            temp_wrong_word = temp_wrong_word + ''
        else:
            temp_wrong_word = temp_wrong_word + word[i]

    if (word == temp_wrong_word):
        temp_wrong_word = ""

    return temp_wrong_word


def replace_with_keyBoard_character(word):
    wrong_word_list = []

    for i in range(1, len(word) - 1):
        if (word[i] >= 'a' and word[i] <= 'z'):

            wrong_values = keyBoardCharacterMapping[word[i]]

            for wrong_value in wrong_values:
                temp_word = word[0:i] + wrong_value + word[i + 1:len(word)]
                wrong_word_list.append(temp_word)

    return wrong_word_list


def add_new_letter(word):
    random_char = random.randrange(97, 97 + 25)
    random_position = random.randrange(1, len(word))

    new_wrong_word = word[0:random_position] + chr(random_char) + word[random_position:len(word)]

    return new_wrong_word


def swap_letter(string):
    i = random.randrange(1, len(string) - 1)
    j = i + 1

    return ''.join((string[:i], string[j], string[i + 1:j], string[i], string[j + 1:]))


if __name__ == '__main__':
    filter_raw_data()

    print("1. Character position data processing")
    print("2. NO Character position data processing")
    print("3. Both system data processing")
    choice = input()

    if (choice == '1'):
        correctWord_to_wrongWord_with_space_and_character_position()
    elif (choice == '2'):
        correctWord_to_wrongWord_with_space_and_no_character_position()
    elif (choice == '3'):
        correctWord_to_wrongWord_with_both_system()

    print("DONE")
