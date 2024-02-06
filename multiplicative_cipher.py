import string
SPACE = string.ascii_uppercase + string.ascii_lowercase + " "
SIZE = len(SPACE)

# def mmi(a):
#     if a == 1:
#         return

#     # compute y to be coprime to x 
#     b = x + 1
#     while x:
#         pass

def mono_enc(key, plaintext):
    return ''.join([SPACE[(SPACE.index(i)*key)%SIZE] for i in plaintext])

def mono_dec(inv_key, cyphertext):
    return ''.join([SPACE[(SPACE.index(i)*inv_key)%SIZE] for i in cyphertext])

def calculate_inverse(key):
    for i in range(SIZE):
        if (i * key) % SIZE == 1: return i

if __name__ == "__main__":
    print("Initializing")
    print("  (Enter 'x' to exit)")
    while 1:
        cmd = input("    Enter E to encrypt, D to decrypt: ").strip().lower()
        if cmd == 'x':
            print("Aborting")
            break
        elif cmd != 'e' and cmd != 'd':
            print("  Invalid command, retry")
        else:
            key = ''
            while key.isnumeric():
                key = input("    Enter a numeric integral key: ")
                if key.isnumeric():
                    key = int(key)
                else:
                    print("  Invalid key, retry")

            if cmd == 'e':
                plaintext = input("    Enter plaintext: ")
                print(f"  Encrypted text: {mono_enc(key, plaintext)}")
            else:
                cyphertext = input("    Enter cyphertext: ")
                inv_key = calculate_inverse(key)
                print(f"  Decrypted text: {mono_dec(inv_key, cyphertext)}")


