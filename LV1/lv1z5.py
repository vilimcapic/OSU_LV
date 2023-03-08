ham_messages = []
spam_messages = []

fhand = open("SMSSpamCollection.txt")

for line in fhand:
    parts = line.split("\t")
    if parts[0] == "ham":
        ham_messages.append(parts[1])
    elif parts[0] == "spam":
        spam_messages.append(parts[1])
fhand.close()

ham_word_count = 0
ham_message_count = len(ham_messages)

for message in ham_messages:
    words = message.split()
    ham_word_count += len(words)

ham_word_avg = ham_word_count / ham_message_count

print("Average number of words in ham: ", ham_word_avg)

spam_word_count = 0
spam_message_count = len(spam_messages)

for message in spam_messages:
    words = message.split()
    spam_word_count += len(words)

spam_word_avg = spam_word_count / spam_message_count

print("Average number of words in spam: ", spam_word_avg)

spam_excl_count = 0

for message in spam_messages:
    if message.strip()[-1] == "!":
        spam_excl_count += 1

print("Number of spam ending with ! ", spam_excl_count)