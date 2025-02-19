from lxml import etree
def determine_time_unit(filepath):
    # Parse the mzML file
    tree = etree.parse(filepath)
    root = tree.getroot()

    # Namespace handling for mzML
    ns = {'ns': 'http://psi.hupo.org/ms/mzml'}

    for spectrum in root.findall('.//ns:spectrum', namespaces=ns):
        for scan in spectrum.findall('.//ns:scan', namespaces=ns):
            for cv_param in scan.findall('.//ns:cvParam', namespaces=ns):
                if cv_param.get('accession') == "MS:1000016":  # Scan start time
                    return(cv_param.get('unitName'))  # Extract unitName
                    break
