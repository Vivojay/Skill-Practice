import string
alphabets = string.ascii_uppercase + string.ascii_lowercase

def mono_enc(key, plaintext):
    plaintext = ''.join([i for i in plaintext if i.isalpha()])
    return ''.join([alphabets[(alphabets.index(i)+key)%52] for i in plaintext])

def mono_dec(key, cyphertext):
    cyphertext = ''.join([i for i in cyphertext if i.isalpha()])
    return ''.join([alphabets[(alphabets.index(i)-key)%52] for i in cyphertext])

"""
if __name__ == "__main__":
    print("Enter 'x' to exit")
    while 1:
        cmd = input("  Enter E to encrypt, D to decrypt: ").strip().lower()
        if cmd == 'x':
            print("Aborting")
            break
        elif cmd != 'e' and cmd != 'd':
            print("Invalid command, retry")
        else:
            key = ''
            while 1:
                key = input("  Enter a numeric integral key: ")
                if key.isnumeric():
                    break
                else:
                    print("Invalid key, retry")

            if cmd == 'e':
                key = int(key)
                plaintext = input("  Enter plaintext: ")
                print(f"Encrypted text: {mono_enc(key, plaintext)}")
            else:
                key = int(key)
                cyphertext = input("  Enter cyphertext: ")
                print(f"Decrypted text: {mono_dec(key, cyphertext)}")
"""

if __name__ == "__main__":
    msg = input("Enter message: ")
    key = int(input("Enter key: "))
    cipher = mono_enc(key, msg)
    print(f"Encrypted msg: {cipher}")
    print(f"Decrypted msg: {mono_dec(key, cipher)}")

