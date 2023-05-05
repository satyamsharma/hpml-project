import logging

# set up logging, if it hasn't been already
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        datefmt='%Y-%m-%d %I:%M:%S',
        level=logging.INFO,
        format='%(asctime)s %(module)s %(levelname)s %(message)s')