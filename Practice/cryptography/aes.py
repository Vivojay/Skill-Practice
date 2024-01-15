# ADVANCED ENCRYPTION STANDARD (AES) - 128 bit

# -- Imports -- #
import os
import hashlib
import numpy as np

# -- Important constants and array definitions -- #
# Lookup Tables (as hex)
FWD_S_BOX = np.array([
    [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76],
    [0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0],
    [0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15],
    [0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75],
    [0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84],
    [0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF],
    [0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8],
    [0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2],
    [0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73],
    [0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB],
    [0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79],
    [0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08],
    [0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A],
    [0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E],
    [0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF],
    [0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]
])

# Inverse Lookup Table (in hex)
INV_S_BOX = np.array([
    [0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB],
    [0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB],
    [0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E],
    [0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25],
    [0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92],
    [0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84],
    [0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06],
    [0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B],
    [0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73],
    [0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E],
    [0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B],
    [0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4],
    [0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F],
    [0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF],
    [0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61],
    [0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D],
])

MIX_MAT = np.array([
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2],
])

INV_MIX_MAT = np.array([
    [0x0e, 0x0b, 0x0d, 0x09],
    [0x09, 0x0e, 0x0b, 0x0d],
    [0x0d, 0x09, 0x0e, 0x0b],
    [0x0b, 0x0d, 0x09, 0x0e],
])

RC = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]

# # User set variables
# CONTENT_TYPE = 'STR' # Leave as it is for string input
# """
# Possible values:
# value: STR | Type of key: String
# value: BIN | Type of key: Binary
# """

# Only implemented flavour as of yet
AES_FLAVOUR = 128

# Flavours not implemented as of yet
# AES_FLAVOUR = 196
# AES_FLAVOUR = 256

AES_INFO = {
    128: {
        "GALOIS_REDUCED_POLY_2P8": 0b100011011, # 283
        "TOTAL_ROUND_COUNT": 10,
    },

    # Unimplemented flavours as of yet
    # 196: {
    #     "GALOIS_REDUCED_POLY_2P8": 0b101110001, # 369
    #     "TOTAL_ROUND_COUNT": 12,
    # },

    # 256: {
    #     "GALOIS_REDUCED_POLY_2P8": 0b101110001, # 369
    #     "TOTAL_ROUND_COUNT": 14,
    # }    
}

TOTAL_ROUND_COUNT = AES_INFO[AES_FLAVOUR]["TOTAL_ROUND_COUNT"]
GALOIS_REDUCED_POLY_2P8 = AES_INFO[AES_FLAVOUR]["GALOIS_REDUCED_POLY_2P8"]
vis_func_block = lambda c: np.resize(np.vectorize(lambda x:f"{x:02x}")(c), (4, 4))

# -- Function Definitions -- #
def gcd(a, b):
    while (a%b != 0):
        # Make sure a > b
        if (a<b):
            a, b = b, a
        # Update a and b
        a, b = b, a%b
        if not b:
            return a
    # Return smaller value (smallest possible factor)
    return b

def rot_arr(arr, rots, direction=0, inplace=True):
    """
    Direction:
    0: Left Rotate
    1: Right Rotate
    """

    if not inplace:
        new_arr = np.copy(arr)

    size = len(arr)
    cycles = gcd(size, rots)

    for i in range(cycles):
        for j in range(size//cycles):
            if j == 0:
                if inplace: t = arr[i]
                else: t = new_arr[i]
                continue
            # Right rotate
            if direction:
                x, y = (i-rots*(j-1))%size, (i-rots*j)%size
            # Left rotate
            else:
                x, y = (i+rots*(j-1))%size, (i+rots*j)%size
            if inplace:
                arr[x], arr[y] = arr[y], arr[x]
            else:
                new_arr[x], new_arr[y] = new_arr[y], new_arr[x]

        if inplace: arr[y] = t
        else: new_arr[y] = t

    if not inplace:
        return new_arr

def sub_val(byte, inverse=False):
    global FWD_S_BOX, INV_S_BOX

    _byte = format(byte, 'b').zfill(8) # Byte size is equal to 8 bits
    if inverse:
        return INV_S_BOX[
            int(_byte[:4], 2),
            int(_byte[4:], 2)
        ]
    else:
        return FWD_S_BOX[
            int(_byte[:4], 2),
            int(_byte[4:], 2)
        ]

def g_func(round_number, word):
    global RC
    rconst = RC[round_number]

    # Circular left shift and subsequent byte substitution
    x = np.vectorize(sub_val)(rot_arr(word, 1, 0, inplace=False))
    # Defining round constant array
    r = np.array([rconst, 0, 0, 0])
    # Adding round constant and returning as output
    return x^r

def galois_mul(a, b, r, degree_of_r=8):
    """
    Degree of any binary number `a` may be calculated
    using the operation: (a >> degree_of_r)

    Using this value, we get the `degree_of_r`th bit from the right.
    If this value is 1, then deg(a) is atleast `degree_of_r`,
    otherwise, deg(a) < `degree_of_r`
    
    If we can ensure/enforce that deg(a) is [always <= `degree_of_r`], - (i)
    then we can directly use the above value as a check.
    i.e. if value is 1, [deg(a) >= `degree_of_r`] - (ii)
    therefore, using eqns (i) and (ii), we can conclude that both degrees must be equal)
    """

    result = 0
    # print(f"a:{a:0{degree_of_r}b} ({a}), b:{b:0{degree_of_r}b} ({b}), res:{result:0{degree_of_r}b} ({result})")
    while (a): # or calculate the exact number of iterations as: math.ceil(math.log(13)/math.log(2))
        # If LSB of `a` is set, add it to result
        result ^= b if (a&1) else 0
        a >>= 1 # Right-shift dividend by 1 position
        b <<= 1 # Left-shift divisor by 1 position
        # Need to reduce poly `b`, if bit #1 (MSB) is set
        b ^= r if (b >> degree_of_r) else 0
        # print(f"a:{a:0{degree_of_r}b} ({a}), b:{b:0{degree_of_r}b} ({b}), res:{result:0{degree_of_r}b} ({result})")

    return result

def get_all_round_keys(key):
    round_count, round_key = 0, np.copy(key)
    # Transposing round_key as required by AES
    rkey = np.resize(np.resize(round_key, (round_key.size//4, 4)).T, round_key.size)
    rkeys = np.array([rkey])
    for round_count in range(1, TOTAL_ROUND_COUNT+1):
        round_key = schedule_next_round_key(round_count, round_key)
        rkey = np.resize(np.resize(round_key, (round_key.size//4, 4)).T, round_key.size)
        rkeys = np.vstack((rkeys, rkey))
    return rkeys

def schedule_next_round_key(round_count, round_key):
    _round_key = np.copy(round_key)

    # round_count must be >= 1
    val = _round_key[:4] ^ g_func(round_count-1, _round_key[12:16])
    _round_key[:4] = val
    for byte in range(len(_round_key)//4 - 1):
        r = _round_key[(byte+1)*4 : (byte+2)*4]
        val ^= r
        _round_key[(byte+1)*4 : (byte+2)*4] = val

    return _round_key

def shift_rows(state_array, inverse=False):
    for i in range(1, len(state_array)//4):
        # not inverse:
        #   if inverse == 0, then rotate left
        #   if inverse == 1, then rotate right
        rot_arr(state_array[i*4 : (i+1)*4], i, direction=inverse)

def mix_cols(mix_mat, state_array):
    global GALOIS_REDUCED_POLY_2P8

    sa = np.resize(state_array, (len(state_array)//4, 4))
    prod = np.array([])
    for i in range(4):
        row = np.array([])
        for j in range(4):
            s = 0
            for x, y in zip(sa[:,i], mix_mat[j]):
                s ^= galois_mul(x, y, GALOIS_REDUCED_POLY_2P8, 8)
            row = np.append(row, s)
        prod = np.append(prod, row)

    return prod.astype(int)
    # np.vectorize(hex)(np.vectorize(int)(state_array))

def generate_key(password,
                 salt,
                 aes_flavour=AES_FLAVOUR,
                 iterations=20):
    assert iterations > 0

    key_len = aes_flavour//8
    if password is None: password = os.urandom(key_len)

    key = hashlib.pbkdf2_hmac('sha256',
                              password,
                              salt,
                              100000,
                              dklen=key_len)
    return key

def enc_file(key, file_path, output_file=None, show_progress=True):
    if output_file is None:
        output_file = file_path

    if not os.path.isfile(file_path):
        print("Invalid file path provided")
        return 1

    try:
        with open(file_path, 'rb') as fp:
            plaintext = fp.read()
        ciphertext = enc_text(key, plaintext, show_progress=show_progress)
        with open(output_file, 'wb') as fp:
            fp.write(ciphertext)
        return 0
    except OSError:
        print("File Error encountered, could not sucessfully encrypt file")
    except Exception:
        print("Some error occured, could not sucessfully encrypt file")

    try:
        with open(file_path, 'wb') as fp:
            fp.write(plaintext)
        print("File has been reverted")
    except Exception:
        print("File could not be reverted")

    return 1

def dec_file(key, file_path, output_file=None, show_progress=True):
    if output_file is None:
        output_file = file_path

    if not os.path.isfile(file_path):
        print("Invalid file path provided")
        return 1

    try:
        with open(file_path, 'rb') as fp:
            ciphertext = fp.read()
        plaintext = dec_text(key, ciphertext, show_progress=show_progress)
        with open(output_file, 'wb') as fp:
            fp.write(plaintext)
        return 0
    except OSError:
        print("File Error encountered, could not sucessfully encrypt file")
    except Exception:
        raise
        print("Some error occured, could not sucessfully encrypt file")

    try:
        with open(file_path, 'wb') as fp:
            fp.write(ciphertext)
        print("File has been reverted")
    except Exception:
        print("File could not be reverted")

    return 1

def enc_text(pwd, content, show_progress=False):
    # matrix visualization lambda function
    global AES_FLAVOUR, MIX_MAT, padded_plaintext, cipher_block, state_array, vis_func_block

    salt_size = AES_FLAVOUR//8
    iterations = 20
    salt = os.urandom(salt_size)
    key = generate_key(password=pwd.encode(), salt=salt, iterations=iterations)
    key = np.array(bytearray(key))

    # if key_type == 'STR':
    #     key = np.array([ord(i) for i in key])

    if isinstance(content, bytes):
        plaintext = list(bytearray(content))
    else:
        plaintext = [ord(i) for i in content]

    ciphertext_blocks = []

    # Add PKCS#7 Padding
    block_size = AES_FLAVOUR//8 # Where '128' is the type/flavour of AES
    pad_len = block_size - len(plaintext)%(block_size)
    padded_plaintext = plaintext + pad_len*[pad_len]
    padded_plaintext = np.array(padded_plaintext)
    block_count = len(padded_plaintext)//block_size

    for i in range(block_count):
        cipher_block = padded_plaintext[block_size*i : block_size*(i+1)]

        # Initial Round
        state_array = np.copy(cipher_block)
        # Transposing state_array as required by AES
        state_array = np.resize(np.resize(state_array, (state_array.size//4, 4)).T, state_array.size)
        round_count, round_key = 0, np.copy(key)
        # Transposing round_key as required by AES
        rkey = np.resize(np.resize(round_key, (round_key.size//4, 4)).T, round_key.size)
        state_array ^= rkey # Add round key

        # Repeating Rounds
        for round_count in range(1, TOTAL_ROUND_COUNT+1):
            # Substitute Bytes
            # np.split(state_array, block_size//4)
            state_array = np.vectorize(sub_val)(state_array)
            # Shift Rows
            shift_rows(state_array)
            if round_count != TOTAL_ROUND_COUNT:
                # Mix Columns
                state_array = mix_cols(MIX_MAT, state_array)
                # Transpose state_array to make it suitable as per AES
                state_array = np.resize(np.resize(state_array, (state_array.size//4, 4)).T, state_array.size)

            # Add Round Key
            round_key = schedule_next_round_key(round_count, round_key)
            rkey = np.resize(np.resize(round_key, (round_key.size//4, 4)).T, round_key.size)
            state_array ^= rkey

            # Uncomment following 6 lines to visualize state arrays and round keys utilized 
            # print(f'round: {round_count}')
            # print('round key')
            # print(vis_func_block(round_key), end='\n\n')
            # print('state arr')
            # print(vis_func_block(state_array))
            # print('-'*80+'\n')

        ciphertext_blocks.append(state_array)
        if show_progress:
            print(f"Encrypted block {i+1}/{block_count} | {round((i+1)/block_count*100, 2)}%", end='\r')

    if show_progress: print()
    # Encrypted text out
    ciphertext_blocks = np.array(ciphertext_blocks).astype(int)

    # if required, uncommented below 4 lines for block visualization
    # print("Encrypted blocks")
    # ciphertext_blocks = np.resize(ciphertext_blocks, (ciphertext_blocks.size//block_size, block_size//4, 4))
    # ciphertext_blocks_vis = np.vectorize(lambda x:f"{x:02x}")(ciphertext_blocks_vis)
    # print(ciphertext_blocks_vis)

    ciphertext = ''
    for cipher_block in ciphertext_blocks:
        cipher_blockT = np.resize(np.resize(cipher_block, (cipher_block.size//4, 4)).T, cipher_block.size)
        cipher_blockT_HEX = np.vectorize(lambda x:f"{x:02x}")(cipher_blockT)
        ciphertext += ''.join(cipher_blockT_HEX)

    return salt + ciphertext.encode()

def dec_text(pwd, ciphertext, show_progress=False):
    # matrix visualization lambda function
    global MIX_MAT, padded_plaintext, cipher_block, state_array, vis_func_block

    # if key_type == 'STR':
    #     key = np.array([ord(i) for i in key])

    salt_size = 16
    iterations = 20

    salt = ciphertext[:salt_size]
    ciphertext = ciphertext[salt_size:]
    key = generate_key(password=pwd.encode(), salt=salt, iterations=iterations)
    key = np.array(bytearray(key))

    try:
        ciphertext = [ciphertext[2*x:2*(x+1)] for x in range(len(ciphertext)//2)]
        ciphertext = np.array([int(i, 16) for i in ciphertext])
    except ValueError:
        print("Invalid ciphertext provided for decryption")
        return None

    plaintext_blocks = []
    rkeys = get_all_round_keys(key)
    block_size = AES_FLAVOUR//8 # Where '128' is the type/flavour of AES
    block_count = len(ciphertext)//block_size

    for block_num in range(block_count):
        # Defining current block
        block = ciphertext[block_size*block_num : block_size*(block_num+1)]

        # Initial Round
        state_array = np.copy(block)
        # Transposing state_array as required by AES
        state_array = np.resize(np.resize(state_array, (state_array.size//4, 4)).T, state_array.size)
        # adding round key (Last round key first)
        state_array ^= rkeys[TOTAL_ROUND_COUNT]

        # Repeating Rounds
        for round_count in range(1, TOTAL_ROUND_COUNT+1):
            # print(f"round: {TOTAL_ROUND_COUNT-round_count+1}, rc: {round_count}")
            # print(vis_func_block(state_array))

            # OPERATION: Inverse Shift Rows
            shift_rows(state_array, inverse=True)
            # print(vis_func_block(state_array))

            # OPERATION: Inverse Substitute Bytes
            # np.split(state_array, block_size//4)
            state_array = np.vectorize(lambda x:sub_val(x, inverse=True))(state_array)
            # print(vis_func_block(state_array))

            # OPERATION: Add Inverse Round Key
            rkey = rkeys[TOTAL_ROUND_COUNT-round_count]
            # rkey = np.resize(np.resize(round_key, (round_key.size//4, 4)).T, round_key.size)
            state_array ^= rkey
            # print(vis_func_block(state_array))

            # OPERATION: Skip mix-cols in last round
            if round_count != TOTAL_ROUND_COUNT:
                # Inverse Mix Columns
                state_array = mix_cols(INV_MIX_MAT, state_array)
                # Transpose state_array to make it suitable as per AES
                state_array = np.resize(np.resize(state_array, (state_array.size//4, 4)).T, state_array.size)

            # print(vis_func_block(state_array))
            # print('-'*80)

        plaintext_blocks.append(state_array)
        if show_progress:
            print(f"Decrypted block {round_count+1}/{block_count} | {round((round_count+1)/block_count*100, 2)}%", end='\r')

    if show_progress: print()
    # Decrypted text out
    plaintext_blocks = np.array(plaintext_blocks).astype(int)
    plaintext_blocksT = [np.resize(np.resize(plain_block, (plain_block.size//4, 4)).T, plain_block.size) for plain_block in plaintext_blocks]

    # Remove PKCS#7 padding characters from the end
    plaintext_blocksT[-1] = plaintext_blocksT[-1][:len(plaintext_blocksT[-1])-plaintext_blocksT[-1][-1]]

    plaintext_blocks = []
    _ = [plaintext_blocks.extend(i) for i in plaintext_blocksT]

    return ''.join([chr(i) for i in plaintext_blocks]).encode() # plaintext

