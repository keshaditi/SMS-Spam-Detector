mkdir -p~/.streamlit/

echo "\
[server]\n\n
port =$port\n\
enableCORS=false\n\
headless=true\n\
\n\
"> /.streamlit/config.toml
