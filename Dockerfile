FROM abhirupghosh1992/stable-diffusion:latest

WORKDIR /home/morpheus

COPY token.txt /home/morpheus

COPY main.py /home/morpheus

EXPOSE 8000

CMD python -m main.py