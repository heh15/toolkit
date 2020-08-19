def finditer_with_line_numbers(pattern, string, flags=0):
    '''
    A version of 're.finditer' that returns '(match, line_number)' pairs.
    '''
    import re

    matches = list(re.finditer(pattern, string, flags))
    if not matches:
        return []

    end = matches[-1].start()
    # -1 so a failed 'rfind' maps to the first line.
    newline_table = {-1: 0}
    for i, m in enumerate(re.finditer(r'\n', string), 1):
        # don't find newlines past our last match
        offset = m.start()
        if offset > end:
            break
        newline_table[offset] = i
   

    # Failing to find the newline is OK, -1 maps to 0.
    matchresults = []
    for m in matches:
        newline_offset = string.rfind('\n', 0, m.start())
        start_number = newline_table[newline_offset]
        rows_number = len(m.group().split('\n'))
        stop_number = start_number + rows_number
        matchresults.append((m, start_number, stop_number))

    return matchresults

    
