list = []

while True:
    input = input()
    if input == "Done":
        break
    try:
        number = float(input)
        list.append(number)
    except ValueError:
        print("Not a number")

if len(list) > 0:
    print("Total numbers ", len(list))
    average = sum(list) / len(list)
    print("Average is ", average)
    print("Minimum ", min(list))
    print("Maximum ", max(list))
    list.sort()
    print(list)
else:
    print("Empty list")