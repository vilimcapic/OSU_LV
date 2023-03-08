word_dictionary = {}
fhand = open ('song.txt')
for line in fhand :
    lineWords = line.rstrip().split()
    for word in lineWords:
        if word in word_dictionary:
            word_dictionary[word] = word_dictionary[word] + 1
        else:
            word_dictionary[word] = 1
fhand.close()

counter = 0
for word in word_dictionary:
    if word_dictionary[word] == 1:
        print(word)
        counter = counter + 1

print('Unique words ', counter)