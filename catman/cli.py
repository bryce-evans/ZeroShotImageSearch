
import argparse
import logging
import sys 
from PIL import Image

from zshot import CatalogManager


def setup_logging(log_level_str):
    """
    Set the log level from the parsed input arg.
    Taken from https://docs.python.org/3/howto/logging.html
    """
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level_str}")
    logging.basicConfig(level=numeric_level)


def create_from_directory(args):
    logging.info(f"Creating catalog from directory {args.dir} ...")
    manager = CatalogManager()
    manager.new_catalog(args.output_path)
    manager.add_directory(args.dir)
    logging.info(f"Found {manager.active_db.count()} images")
    manager.write_to_disk(args.output_path)
    logging.info(f"Wrote catalog to {args.output_path}.")

    
def search(args):
    logging.info(f"Searching catalog {args.catalog} for {args.query} ...")
    manager = CatalogManager()
    manager.load_from_disk(args.catalog)

    logging.info(f"Found {manager.active_db.count()} images in loaded catalog.")
    image_paths = manager.search(args.query, count=args.count)

    for path in image_paths:
        # TODO: handle displaying remote images.
        if path.startswith("https://"):
            logging.warning("Remote path matched. Can't display.")
            pass

        

        image = Image.open(path)

        # TODO: find a better way to show images.
        # This causes the process to be forked, which
        # throws warnings from huggingface libraries
        image.show()



def main(args):
    parser = argparse.ArgumentParser(description="CatMan: Catalog Manager for Fast Image Search")
    parser.add_argument('--log', default='warning', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='Set the logging level (default: %(default)s)')
    
    args, remaining_argv = parser.parse_known_args()
    setup_logging(args.log)

    subparsers = parser.add_subparsers()

    # Create from directory
    parser_add = subparsers.add_parser('create', help='Loads all jpgs in a directory and creates a catalog')
    parser_add.add_argument('dir', type=str, help='input directory')
    parser_add.add_argument('output_path', type=str, help='output directory')
    parser_add.set_defaults(func=create_from_directory)

    # Query catalog
    parser_add = subparsers.add_parser('search', help='Searches a catalog for the closest images that match query')
    parser_add.add_argument('catalog', type=str, help='catalog path')
    parser_add.add_argument('query', type=str, help='query to search for')
    parser_add.add_argument('--count', required=False, type=int, default=1, help='max number of images to return')
    parser_add.set_defaults(func=search)

    args = parser.parse_args(remaining_argv)
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])