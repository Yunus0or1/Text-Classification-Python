import nltk
from nltk.tokenize import word_tokenize

sentence = "Huge Jam from Banani to Gulshan"

words = word_tokenize(sentence)


jam_place_integer = 0  # This will be needed for first and second phase
jam_place2_integer = 0  # This will be needed for third phase
jam_place3_integer = 0  # This will be needed for third phase because 'to' will have two places. The before and after there will be two place names.
enter_to_second_phase = 0
enter_to_third_phase = 0
enter_to_fourth_phase = 0

if 'near' in words or 'at' in words or 'of' in words:
    for x in range(0, len(words)):
        if words[x] == 'near' or words[x] == 'at' or words[x] == 'of':
            jam_place_integer = x + 1
else:
    enter_to_second_phase = 1

if enter_to_second_phase == 1:
    if 'in' in words:
        for x in range(0, len(words)):
            if words[x] == 'in':
                if (str(words[x + 1]) == 'traffic' or str(words[x + 1]) == 'jam'
                        or str(words[x + 1]) == 'grid' or str(
                        words[x + 1]) == 'lock'):
                    enter_to_third_phase = 1
                else:
                    pos_tag_of_next_word = nltk.pos_tag(word_tokenize(words[x + 1]))
                    word, tag = zip(*pos_tag_of_next_word)
                    pos_tag_of_next_word_str = str(''.join(tag))
                    if pos_tag_of_next_word_str == 'NN' or pos_tag_of_next_word_str == 'NNP':
                        jam_place_integer = x + 1
    else:
        enter_to_third_phase = 1

if enter_to_third_phase == 1:
    if 'to' in words:
        for x in range(0, len(words)):
            if words[x] == 'to':
                jam_place2_integer = x + 1
                # Now finding if previous word is a place also
                pos_tag_of_previous_word = nltk.pos_tag(word_tokenize(words[x - 1]))
                word, tag = zip(*pos_tag_of_previous_word)
                pos_tag_of_previous_word_str = str(''.join(tag))
                if pos_tag_of_previous_word_str == 'NN' or pos_tag_of_previous_word_str == 'NNP':
                    jam_place3_integer = x - 1

    else:
        enter_to_fourth_phase = 1

jam_place_final_result = ''


if jam_place_integer != 0:
    jam_place = words[jam_place_integer]
    jam_place_final_result = jam_place


if enter_to_third_phase == 1 and enter_to_fourth_phase == 0:
    if jam_place3_integer ==0:
        jam_place = words[jam_place2_integer]
        jam_place_final_result =  jam_place
    if jam_place3_integer !=0:
        jam_place2 = words[jam_place2_integer]
        jam_place3 = words[jam_place3_integer]
        jam_place_final_result =  jam_place3 + ' to ' + jam_place2


print(jam_place_final_result)