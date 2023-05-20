def shuffle(arr_in, arr_out, seed):
    for item in arr_in:
        arr_out.append(item)

    return arr_out


arr_in = [0, 1, 2, 3]
arr_out = []

seed = 0
shuffle(arr_in, arr_out, seed)

print(arr_out)
