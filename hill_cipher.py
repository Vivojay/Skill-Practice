import numpy as np

def encrypt(key, msg):
    msg = ''.join([i for i in msg if i.isalpha()]).upper()
    key = ''.join([i for i in key if i.isalpha()]).upper()

    to_mat = lambda x: np.vectorize(ord)(np.array(list(x))).astype(np.int32) - 65
    to_chars = lambda x: np.array(list(x)).astype(np.int32) + 65
    key_len_sqrt = len(key)**(1/2)

    # Pad key
    if key_len_sqrt.is_integer():
        key_len_sqrt = int(key_len_sqrt)
    else:
        key_len_sqrt = int(key_len_sqrt) + 1
        padding_len = key_len_sqrt**2 - len(key)
        key += ''.join([chr(i%26 + 65) for i in range(padding_len)])

    key_matrix = np.reshape(to_mat(key), (key_len_sqrt, key_len_sqrt))
    msg_matrix = to_mat(msg)

    if msg_matrix.size % key_len_sqrt:
        # pad all remaining positions of msg with 'z'
        padding_char = 25
        padding = np.array([padding_char]*(key_len_sqrt - msg_matrix.size%key_len_sqrt))
        msg_matrix = np.hstack((msg_matrix, padding))

    msg_len = msg_matrix.size
    msg_matrix = np.split(msg_matrix, msg_len//key_len_sqrt)

    res = []
    for msg_part in msg_matrix:
        msg_part = np.reshape(msg_part, (msg_part.size, 1))
        enc_mat = np.vectorize(round)(np.dot(key_matrix, msg_part) % 26)
        enc_mat = np.reshape(enc_mat, enc_mat.size)
        enc_mat = np.vectorize(chr)(to_chars(enc_mat))
        res.append(''.join(enc_mat))

    return ''.join(res)

def decrypt(key, cipher):
    key = ''.join([i for i in key if i.isalpha()]).upper()
    cipher = ''.join([i for i in cipher if i.isalpha()]).upper()

    to_mat = lambda x: np.vectorize(ord)(np.array(list(x))).astype(np.int32) - 65
    to_chars = lambda x: np.array(list(x)).astype(np.int32) + 65
    key_len_sqrt = len(key)**(1/2)
    cipher_len = len(cipher)

    # Pad key
    if key_len_sqrt.is_integer():
        key_len_sqrt = int(key_len_sqrt)
    else:
        key_len_sqrt = int(key_len_sqrt) + 1
        key += ''.join([chr(i%26 + 65) for i in range(key_len_sqrt**2 - len(key))])

    key_matrix = np.reshape(to_mat(key), (key_len_sqrt, key_len_sqrt))
    det_mod = round(np.linalg.det(key_matrix) % 26)
    det_inv = pow(det_mod, -1, 26)
    adj_mod = (np.linalg.inv(key_matrix) * np.linalg.det(key_matrix)) % 26
    mat_mod_inv = (adj_mod * det_inv) % 26

    cipher_matrix = np.split(to_mat(cipher), cipher_len//key_len_sqrt)

    res = []
    for cipher_part in cipher_matrix:
        cipher_part = np.reshape(cipher_part, (cipher_part.size, 1))
        dec_mat = np.vectorize(round)(np.dot(mat_mod_inv, cipher_part)) % 26
        dec_mat = np.reshape(dec_mat, dec_mat.size)
        dec_mat = np.vectorize(chr)(to_chars(dec_mat))
        res.append(''.join(dec_mat))

    return ''.join(res)

if __name__ == "__main__":
    msg = input("Enter message: ")
    key = input("Enter key: ")
    cipher = encrypt(key, msg)
    print(f"Encrypted msg: {cipher}")
    print(f"Decrypted msg: {decrypt(key, cipher)}")


