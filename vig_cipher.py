def encrypt(key, msg):
    key = ''.join([i for i in key if i.isalpha()]).upper()
    msg = ''.join([i for i in msg if i.isalpha()]).upper()
    msg_len, key_len = len(msg), len(key)
    assert len(msg) >= len(key)

    padding_len = msg_len - key_len
    key += ''.join([key[i%key_len] for i in range(padding_len)])

    res = [
        chr(((ord(msg[i]) - 65 + ord(key[i]) - 65) % 26) + 65)
        for i in range(msg_len)
    ]

    return ''.join(res)

def decrypt(key, cipher):
    key = ''.join([i for i in key if i.isalpha()]).upper()
    cipher = ''.join([i for i in cipher if i.isalpha()]).upper()
    cipher_len, key_len = len(cipher), len(key)
    assert len(cipher) >= len(key)

    padding_len = cipher_len - key_len
    key += ''.join([key[i%key_len] for i in range(padding_len)])

    res = [
        chr(((ord(cipher[i]) - 65 - ord(key[i]) + 65) % 26) + 65)
        for i in range(cipher_len)
    ]

    return ''.join(res)

if __name__ == "__main__":
    msg = input("Enter message: ")
    key = input("Enter key: ")
    cipher = encrypt(key, msg)
    print(f"Encrypted msg: {cipher}")
    print(f"Decrypted msg: {decrypt(key, cipher)}")

