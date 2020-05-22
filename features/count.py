
while input().find("Answer") == -1:
    pass
modelstr = input()
model = modelstr.split(" ")
print("Model:")
for s in model:
    print(s)
print("Model size: {}".format(len(model)))