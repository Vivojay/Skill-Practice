# Importing libraries
import rich
import string
import numpy as np

# Constant definitions
LOWERS = string.ascii_lowercase

# Function definitions
def encrypt(key, plaintext, verbose=0):
    global PLAYFAIR_BOARD
    PLAYFAIR_BOARD = generate_key_table(key)

    if verbose:
        print("Key Table")
        for i in PLAYFAIR_BOARD[:len(key)//5]: rich.print('[red]'+'  '.join(i)+'[/red]')
        i = PLAYFAIR_BOARD[len(key)//5]
        pre, post = i[:len(key)%5], i[len(key)%5:]
        pre, post = '  '.join(pre), '  '.join(post)
        rich.print(f"[red]{pre}  [/red]{post}")
        for i in PLAYFAIR_BOARD[len(key)//5+1:]:
            print('  '.join(i))
        print()

    digraph = process_plaintext(plaintext, verbose=verbose)
    ind = lambda x: (np.where(PLAYFAIR_BOARD==x)[0][0], np.where(PLAYFAIR_BOARD==x)[1][0])
    index_pairs = [(ind(i), ind(j)) for i, j in digraph]

    res = ''

    for ip in index_pairs:
        if ip[0][0] == ip[1][0]:
            # same row
            p = PLAYFAIR_BOARD[ip[0][0], (ip[0][1] + 1) % 5]
            q = PLAYFAIR_BOARD[ip[1][0], (ip[1][1] + 1) % 5]
        elif ip[0][1] == ip[1][1]:
            # same col
            p = PLAYFAIR_BOARD[(ip[0][0] + 1) % 5, ip[0][1]]
            q = PLAYFAIR_BOARD[(ip[1][0] + 1) % 5, ip[1][1]]
        else:
            p = PLAYFAIR_BOARD[ip[0][0], ip[1][1]]
            q = PLAYFAIR_BOARD[ip[1][0], ip[0][1]]

        if verbose: rich.print(f"[red]{PLAYFAIR_BOARD[ip[0]]} {PLAYFAIR_BOARD[ip[1]]}[/red] --> [blue]{p} {q}[/blue]")
        res += p+q

    if verbose: print(f"Encrypted string: {res}")
    return res

def decrypt(key, ciphertext, verbose=0):
    global PLAYFAIR_BOARD
    PLAYFAIR_BOARD = generate_key_table(key)

    if verbose:
        print("Key Table")
        for i in PLAYFAIR_BOARD[:len(key)//5]: rich.print('[red]'+'  '.join(i)+'[/red]')
        i = PLAYFAIR_BOARD[len(key)//5]
        pre, post = i[:len(key)%5], i[len(key)%5:]
        pre, post = '  '.join(pre), '  '.join(post)
        rich.print(f"[red]{pre}  [/red]{post}")
        for i in PLAYFAIR_BOARD[len(key)//5+1:]:
            print('  '.join(i))
        print()

    digraph = process_plaintext(ciphertext, verbose=verbose)
    ind = lambda x: (np.where(PLAYFAIR_BOARD==x)[0][0], np.where(PLAYFAIR_BOARD==x)[1][0])
    index_pairs = [(ind(i), ind(j)) for i, j in digraph]

    res = ''

    for ip in index_pairs:
        if ip[0][0] == ip[1][0]:
            # same row
            p = PLAYFAIR_BOARD[ip[0][0], (ip[0][1] - 1) % 5]
            q = PLAYFAIR_BOARD[ip[1][0], (ip[1][1] - 1) % 5]
        elif ip[0][1] == ip[1][1]:
            # same col
            p = PLAYFAIR_BOARD[(ip[0][0] - 1) % 5, ip[0][1]]
            q = PLAYFAIR_BOARD[(ip[1][0] - 1) % 5, ip[1][1]]
        else:
            p = PLAYFAIR_BOARD[ip[0][0], ip[1][1]]
            q = PLAYFAIR_BOARD[ip[1][0], ip[0][1]]

        if verbose: rich.print(f"[blue]{PLAYFAIR_BOARD[ip[0]]} {PLAYFAIR_BOARD[ip[1]]}[/blue] --> [red]{p} {q}[/red]")
        res += p+q

    if verbose: rich.print(f"Decrypted string with fillers: [red]{res}[/red]")

    sans_filler = [res[0]]+[res[i] for i in range(1, len(res)-1) if not (res[i-1] == res[i+1] and res[i] == 'x')]
    if res[-1] != 'x': sans_filler += [res[-1]]
    res = ''.join(sans_filler)

    if verbose: rich.print(f"Final decrypted string: [blue]{res}[/blue]")
    return res

def process_plaintext(plaintext, verbose=0):
    plaintext = ''.join([i for i in plaintext if i.isalpha()])
    plaintext = plaintext.strip().lower()
    res = []

    print(f"pt: {plaintext}")
    for i in range(len(plaintext)//2):
        pair = plaintext[2*i : 2*(i+1)]
        if pair[0] == pair[1]:
            pair = 'x'.join(pair)
            # pair = f"{pair[0]}x{pair[1]}"
        res.append(pair)

    res = ''.join(res)
    print(f"res: {res}")

    if len(plaintext)%2:
        if res[-1] == plaintext[-1]: res += 'x'
        res += plaintext[-1]

    if len(res)%2:
        char = 'z'
        if res[-1] == 'z': char = 'x'
        res += char

    digraph = np.reshape(np.array(list(res)), (len(res)//2, 2) )

    if verbose:
        log_string = ''
        for i, j in enumerate(digraph):
            col = ['red', 'white'][i%2]
            log_string += ' '.join([f'[{col}]' + ''.join(j) + f'[/{col}]']) + ' '
        rich.print(f"[blue]digraphs[/blue]: {log_string}")

    return digraph

def generate_key_table(key):
    global LOWERS
    lowers = LOWERS

    key = ''.join([i for i in key if i.isalpha()]).lower()
    key_distinct = []

    for k in key:
        present: bool = False
        for j in key_distinct:
            if j == k:
                present = True
                break
        if not present: key_distinct.append(k)

    for l in lowers:
        present: bool = False
        for j in key_distinct:
            if j == l:
                present = True
                break
        if not present: key_distinct.append(l)
    
    l = len(key_distinct)
    for i in range(l):
        c = key_distinct[l - i - 1]
        if c == 'i' or c == 'j':
            del key_distinct[l - i - 1]
            break

    l = len(key_distinct)
    for i in range(l):
        c = key_distinct[i]
        if c == 'j':
            key_distinct[i] = 'i'
            break

    key = np.reshape(key_distinct, (5, 5))
    return key

def get_indices(PLAYFAIR_BOARD, digraph):
    # Actual logic for finding each index
    for p, q in digraph:
        inds = [None, None]
        for i in range(len(PLAYFAIR_BOARD)):
            for j in range(len(PLAYFAIR_BOARD[0])):
                c = PLAYFAIR_BOARD[i][j]
                if isinstance(c, str):
                    if c == p:
                        inds[0] = (i, j)
                    if c == q:
                        inds[1] = (i, j)
                else:
                    if c[0] == p or c[1] == p:
                        inds[0] = (i, j)
                    elif c[0] == q or c[1] == q:
                        inds[1] = (i, j)

    return inds
"""
if __name__ == "__main__":
    # Examples
    key = 'playfair example'
    plaintext = "HI DE TH EG OL DI NT HE TR EX ES TU MP"
    verbose = 1

    enc_text = encrypt(key, plaintext, verbose)
    dec_text = decrypt(key, enc_text, verbose)

"""

if __name__ == "__main__":
    verbose = 1
    msg = input("Enter message: ")
    key = input("Enter key: ")

    msg, key = "Haddie is smart", "Get a New Hobby"
    # msg, key = "baby making", "count me up"
    cipher = encrypt(key, msg, verbose=verbose)
    print(f"Encrypted msg: {cipher}")
    print(f"Decrypted msg: {decrypt(key, cipher, verbose=verbose)}")

