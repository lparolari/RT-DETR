import argparse


def main():
    parser = argparse.ArgumentParser(description="Process log files")
    parser.add_argument("log_file", type=str, help="Path to log file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )
    args = parser.parse_args()

    log_file = open(args.log_file, "r")

    log_lines = log_file.readlines()
    log_lines = [line.strip() for line in log_lines]

    log_file.close()

    # keep last logged stats
    i_last = list_rindex(log_lines, "IoU metric: bbox")

    log_lines = log_lines[i_last:]

    # filter out lines that do not contain AP or AR
    log_lines = [
        line
        for line in log_lines
        if "Average Precision" in line or "Average Recall" in line
    ]

    # keep only the last 5 characters of each line with the value of AP or AR
    log_stats = [line[-5:] for line in log_lines]

    # join the stats into a single string
    stats = ",".join(log_stats)

    print(stats)

    if args.verbose:
        for line in log_lines:
            print(line)


def list_rindex(lst, item):
    """
    Find first place item occurs in list, but starting at end of list.
    Return index of item in list, or -1 if item not found in the list.
    """
    i_max = len(lst)
    i_limit = -i_max
    i = -1
    while i >= i_limit:
        if lst[i] == item:
            return i_max + i
        i -= 1
    return -1


if __name__ == "__main__":
    main()
