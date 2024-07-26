from dotenv import load_dotenv
from flask import Flask, render_template, request, session
from flask_session import Session
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    def print_query(result):
        print("Generated SQL Query:", result)
        return result
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
        | print_query
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = [
            AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database.", role="AI"),
        ]
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['user_query']
    
    if 'chat_history' not in session:
        session['chat_history'] = [
            AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
        ]

    session['chat_history'].append(HumanMessage(content=user_query, role="user"))
    
    
    db = init_database(
        user="testenv",
        password="kjoGq6kqswhBX0hY2mz9",
        host="studentdashboard-test.cwxkglzyjyas.us-west-1.rds.amazonaws.com",
        port="3306",
        database="student_dashboard"
    )
    
    response = get_response(user_query, db, session['chat_history'])
    session['chat_history'].append(AIMessage(content=response))
    
    return render_template('index.html', chat_history=session['chat_history'])

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=True)
