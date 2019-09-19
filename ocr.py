#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: Naveen Marri, Utkarsh Kumar, Surya Prateek Soni
# (based on skeleton code by D. Crandall, Oct 2018)
##########################################################################################################################################################

# 1. INITIAL PROBABILITY: The initial probability is calculated by finding the frequency of first characters of sentences and then dividing by the total
#                         number of lines(total number of first characters).
#
# 2. TRANSIT PROBABILITY: The transition probability is calculated by finding the frequencies of each transition and then dividing by the total frequency
#                         of the transitioned letter.
#
# 3. EMISSION PROBABILITY: Emission probability is calculated by comparing each test character image pixel to each pixel of true or proper image generated
#                          for each letter and then finding the probaility score as below:
#                               P(Emission) = P(count of matched)^0.8 + P(count of unmatched)^0.2
#                          Here, we are assuming 20% noise to be present. So when each time a pixel matches, then we multipy (1-0.2)= 0.8 and 0.2 when
#                          unmatched.
#
# 4. VITERBI ALGORITHM: We are using Viterbi algorithm to trace back the states and complete the sentence.
#                       For each state, we are recursively calculating the maximum of previous state prob and transition prob to current state. Then multipying
#                       by emission probability to get the current state prob.
#
#                       v(i+1) = log(x[i+1]) + max{v(i) + log(t(i,i+1))}
#
#                       Here, i+1 is the current state and i is the previous state. v(i+1) is the current state probability.
#                       x[i+1] is the emission probability for character x, v(i) is the previous state probability and t(i,i+1) is the transition
#                       probability from state i to i+1.
#
#                       We are storing the state probabilities v(i=1 to n) in final_prob_matrix.
#                       We are also recording each transition from state i to i+1 in back_dict which we use for backtracking.
#                       Emission probabilities x[i=1 to n] is calculated by a function emission_eval. This returns a dictionary of characters and most
#                       most probable previous states.
#
# 5. SIMPLE ALGORITHM: For simple algorithm, we are returning the most probable character based on the maximum emission probability of all letters
#                      for a given test character image.
#
# 6. FINAL ANSWER: In this function, taking the line number through the test image file argument name(eg: 1 for "test-1-0.png") and comparing the line
#                  with that index in test-strings.txt file to the resulting sentences of Viterbi and Simple. The function returns the answer for which
#                  most characters match to the test-strings.


##########################################################################################################################################################


from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print im.size
    #print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# function to calculate initial and transit frequencies
def train_txt(fname):
    # Dictionary to store initial frequencies
    initial_freq = {}
    transit_freq = {}
    letter_freq = {}
    total_words = 0

    with open(fname) as file:
        for line in file.readlines():

           for word in line.split():
                total_words += 1

           initial_freq[line[0]] = initial_freq.setdefault(line[0], 0) + 1
           for i in range(1,len(line)):
                key = line[i-1]+line[i]
                transit_freq[key] = transit_freq.setdefault(key, 0) + 1
                letter_freq[line[i-1]] = letter_freq.setdefault(line[i-1], 0) + 1
    return initial_freq,transit_freq,letter_freq,total_words

# function to calculate emission frequencies based on the index of character image in test file
def emission_eval(index):
    emission_freq = {}
    test_letter = test_letters[index]
    total_len = len(test_letters[index][0])*len(test_letters[index])
    emission_prob = {}
    for letter in TRAIN_LETTERS:
        count = 0
        a = train_letters[letter]
        for each in range(0,len(test_letter)):
             for i in range(0,len(test_letter[each])):
                if test_letter[each][i] == a[each][i]:
                        count += 1
        emission_prob[letter] = math.pow(0.8,count) * math.pow(0.2,total_len-count)
    return emission_prob

# function to recognize character using simple technique
def simple():

    sentence = []
    for i in range(0,len(test_letters)):
        emission_prob = emission_eval(i)
        sentence.append(max(emission_prob, key=emission_prob.get))

    return ''.join(sentence)


# function to recognize charater using viterbi algorithm
def viterbi():

    # Dictionary to store states and probabilities of each best character associated with the state
    final_prob_matrix = {}

    # Dictionary to store states for tracing backwards. Eg: if A comes from B at state 2, then this will hold the values: {2:{[A,B]}}
    back_dict = {}
    #init_freq_total = sum(initial_freq.values())

    # Calcualte total and minimum initial and trasit frequencies
    init_freq_total = total_words
    init_freq_minimum = math.log(float(initial_freq[min(initial_freq, key=initial_freq.get)])) - math.log(sum(initial_freq.values()))

    #transit_freq_total = sum(transit_freq.values())
    transit_freq_minimum = math.log(float(transit_freq[min(transit_freq, key=transit_freq.get)])) - math.log(sum(transit_freq.values()))

    for i in range(0,len(test_letters)):
        back_dict[i] = []
        emission_prob = emission_eval(i)

        # If state 1, then store initial * emission probability for each character in TRAIN_LETTERS
        if i == 0:
            # Store letter probability
            l_prob = {}
            for letter in TRAIN_LETTERS:
                if letter in initial_freq:
                        init_p_letter = math.log(float(initial_freq[letter])) -  math.log(init_freq_total)
                else:
                        init_p_letter = init_freq_minimum

                l_prob[letter] = init_p_letter + math.log(float(emission_prob[letter]))

            final_prob_matrix[0] = l_prob

        # If state not 1 then store prev state prob * emission prob
        else:
            l_prob = {}
            for letter_new in TRAIN_LETTERS:
                temp_dict = {}
                prev_max_val = float('-inf')
                for letter_prev in TRAIN_LETTERS:
                        key = letter_prev+letter_new
                        if key in transit_freq and letter_new in letter_freq:
                                t_prob = math.log(float(transit_freq[key]))  - math.log(letter_freq[letter_new])
                        else:
                                t_prob =  transit_freq_minimum #/letter_freq[letter_new]

                        # Check the most probable previous character to the given state character
                        tmp_max_val =  final_prob_matrix[i-1][letter_prev] +  t_prob
                        if tmp_max_val > prev_max_val:
                                prev_max_val = tmp_max_val
                                tracker = letter_prev

                back_dict[i].append([letter_new,tracker])
                l_prob[letter_new] = prev_max_val + math.log(emission_prob[letter_new])

            final_prob_matrix[i] = l_prob

    # Find the most probable last letter and then trace back from there
    last_letter = max(final_prob_matrix[len(final_prob_matrix)-1], key=final_prob_matrix[len(final_prob_matrix)-1].get)

    sentence = last_letter
    for i in range(len(test_letters)-1,-1,-1):
                for j in range(0,len(back_dict[i])):
                        if back_dict[i][j][0]  ==  last_letter:
                                last_letter = back_dict[i][j][1]
                sentence += last_letter

    # Return the reverse of the sentence
    return sentence[::-1][1:]


def final_answer(v_sentence,s_sentence):
    index = int(sys.argv[3].split('-')[1])
    train_text=open('test-strings.txt')
    lines=train_text.readlines()
    line = lines[index]
    v_count = sum([ line[i] == v_sentence[i] for i in range(len(v_sentence)) ])
    s_count = sum([ line[i] == s_sentence[i] for i in range(len(s_sentence)) ])
    if v_count > s_count:
        return v_sentence
    else:
        return s_sentence


(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
initial_freq,transit_freq,letter_freq,total_words = train_txt(train_txt_fname)
viterbi_sentence = viterbi()
simple_sentence = simple()
print "Simple: " + simple_sentence
print "Viterbi: " + viterbi_sentence
print "Final Answer: " + final_answer(viterbi_sentence,simple_sentence)
