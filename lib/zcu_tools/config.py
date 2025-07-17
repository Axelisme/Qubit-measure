from argparse import Namespace

config = Namespace(
    # The IP address of local host
    LOCAL_IP="localhost",
    # The port number of local host
    LOCAL_PORT=0,
    # proxy setting
    ONLY_PROXY_DECIMATED=True,
    # Whether to run data server in dry run mode
    DATA_DRY_RUN=False,
    # Whether to run Yoko device in dry run mode
    YOKO_DRY_RUN=False,
    # Labber API directory
    LABBER_API_DIR="./labber_api",
)
