from config import parser
from run.train_ea import train_ea


if __name__ == '__main__':
    args = parser.parse_args()
    train_ea(args)
