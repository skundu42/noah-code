def parse_pages(value: str) -> list[int]:
    pages = []
    for item in value.split(","):
        if "-" in item:
            start, end = item.split("-")
            pages.extend(range(int(start), int(end)))
        else:
            pages.append(int(item))
    return pages
