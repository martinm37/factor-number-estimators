


test_dict = {
    "a" : 2,
    "b" : 7
}

ee = test_dict["a"]

def sum_fun(a,b):
    return a + b

def minus_fun(a,b):
    return a - b

def complicated_fun(a,b,c):
    return a * (b + c)

fun_dict = {
    "sum" : sum_fun,
    "minus" : minus_fun,
    "other_mins" : minus_fun,
    "hmmm" : complicated_fun

}

x = fun_dict["sum"](7,3)
y = fun_dict["minus"](3,8)
z = fun_dict["other_mins"](2,11)
e = fun_dict["hmmm"](3,8,8)
#eee = fun_dict["hmmm"](3,8)


print("eyoooo")