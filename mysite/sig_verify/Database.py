import sqlite3
import os
from IPython.core.display import Image

def Add_Entry(account_number, picture_file_url):
	def create_or_open_db(db_file):
		db_is_new  = not os.path.exists(db_file)
		connection = sqlite3.connect(db_file)
		if db_is_new :
			sql = '''create table if not exists USERS(
			ACCOUNT_NUMBER INTEGER PRIMARY KEY UNIQUE NOT NULL,
			SIGNATURE_URLS TEXT NOT NULL);'''
			connection.execute(sql)
		return connection
		
	connection = create_or_open_db('signatures_dbs.sqlite')
	sql = ''' INSERT INTO USERS(ACCOUNT_NUMBER, SIGNATURE_URLS) VALUES(?,?);'''
	connection.execute(sql,[account_number, picture_file_url])
	connection.commit()

def Extract(account_number):
	def create_or_open_db(db_file):
		db_is_new  = not os.path.exists(db_file)
		connection = sqlite3.connect(db_file)
		if db_is_new :
			sql = '''create table if not exists USERS(
			ACCOUNT_NUMBER INTEGER PRIMARY KEY UNIQUE NOT NULL,
			SIGNATURE_URLS TEXT NOT NULL);'''
			connection.execute(sql)
		return connection
	conn = create_or_open_db('signatures_dbs.sqlite')
	cursor = conn.cursor()
	sql = "SELECT SIGNATURE_URLS FROM USERS WHERE ACCOUNT_NUMBER = :ACCOUNT_NUMBER"
	param = {'ACCOUNT_NUMBER': account_number}
	cursor.execute(sql, param)
	signature_url = cursor.fetchone()
	conn.commit()
	conn.close()
	return (signature_url)
		