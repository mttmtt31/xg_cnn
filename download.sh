# Set the URL of the Google Drive file
id="1EKE3LvTQEt6ppQPi38wl9Inb5fqzBEvz"
destination_path="images.zip"

echo "Downloading and extracting pictures"
# Download the file
gdown $id -O $destination_path

# Extract the tar.gz file
unzip -q $destination_path

# Clean up the downloaded tar.gz file
rm $destination_path

# Set the URL of the Google Drive file
id="16krAXoU-IaF0TLWA34wM8hGzK79jkXVM"
destination_path="dataset.zip"
extracted_folder="data"

echo "Downloading tensors"
# Download the file
gdown $id -O $destination_path

# Extract the tar.gz file
unzip -q $destination_path -d $extracted_folder