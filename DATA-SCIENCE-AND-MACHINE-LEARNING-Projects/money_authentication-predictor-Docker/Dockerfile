# Use the latest Miniconda image, which includes Conda
FROM continuumio/miniconda3:latest  

# Set the working directory
WORKDIR /usr/app

# Copy all files from the current directory to the container
COPY . .

# Update Conda to the latest version
RUN conda update -n base -c defaults conda -y

# Ensure pip is installed and up to date
RUN conda install -y pip && pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "flask_api.py"]



# # Old code form krish naik docker video
# FROM continuumio/anaconda34.4.0
# COPY . /usr/app/
# EXPOSE 5000
# WORKDIR /usr/app/
# RUN pip install -r requirements.txt
# CMD python flask_api.py
