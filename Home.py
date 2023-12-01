import streamlit as st

st.set_page_config(
    page_title="ANOTE Financial Chatbot",
    page_icon="👮‍♂️",
)
st.header( "ANOTE Financial Chatbot :speech_balloon:")

# Create three columns 
col1, col2 = st.columns([1,1])

with col1:
    st.image('images/covid-icon.png')
    st.write('<a href="/covid"> Check out my Covid Dashboard</a>', unsafe_allow_html=True)
    
    #st.markdown ('[![Foo](http://www.google.com.au/images/nav_logo7.png)](http://google.com.au/)')
    link_to_covid_page = ''

    #st.markdown('See my covid dashboard here.')

    st.image('images/friends.png')
    st.write('<a href="https://www.behance.net/datatime">View more pretty data visualizations.</a>', unsafe_allow_html=True)


with col2:
    st.image('images/covid-map.png')
    st.write('<a href="/map"> Check out my Interactive Map</a>', unsafe_allow_html=True)    
    
    st.image('images/github.png')
    st.write('<a href="https://github.com/zd123"> View more awesome code on my github.</a>', unsafe_allow_html=True)