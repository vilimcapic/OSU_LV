
try:
    x = input()
    number = float(x)
    if((number>1.0 or number<0)):
        print('Out of bounds')
    elif(number >= 0.9):
        print('A')
    elif(number >= 0.8):
        print('B')
    elif(number >= 0.7):
        print('C')
    elif(number >= 0.6):
        print('D')
    else:
        print('F')
except ValueError:
    print("Not a number")

    


