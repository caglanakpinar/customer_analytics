"""
Inspiration -> https://www.vitoshacademy.com/hashing-passwords-in-python/
"""

import hashlib
import binascii
import os


def hash_pass( password ):
    """
    Hash a password for storing.
    """
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwd_hash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt, 100000)
    pwd_hash = binascii.hexlify(pwd_hash)
    return (salt + pwd_hash) # return bytes


def verify_pass(provided_password, stored_password):
    """V
    verify a stored password against one provided by user
    """
    stored_password = stored_password.decode('ascii')
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwd_hash = hashlib.pbkdf2_hmac('sha512', provided_password.encode('utf-8'),  salt.encode('ascii'),  100000)
    pwd_hash = binascii.hexlify(pwd_hash).decode('ascii')
    return pwd_hash == stored_password

