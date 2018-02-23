##--------------Backup Subroutines (Remove once others have been checked)----------------


def svgToCoords_bk(svg_path):
	trunc = os.path.splitext(svg_path)[0]
	txt_path = trunc + ".txt"

	shutil.copyfile(svg_path, txt_path)

	in_path = txt_path
	out_path = trunc + "_coordinates.txt"

	in_f = open(in_path, 'r')
	out_f = open(out_path, 'w')

	f = in_f.readlines()

	#type_check = re.compile(r"\<(<type>\w[4])=.*"
	#path_pattern = re.compile(r"\<(<type>\w+) d="M 545.904775,351.973887 A620.572292,620.572292 0.000000 0,1 547.188710,380.040344" fill="none" stroke ="red" stroke-width="1" />"
	line_pattern = re.compile(r'<(?P<type>\w{4}) x1=\"(?P<x1>\d+\.?\d*)\" y1=\"(?P<y1>\d+\.?\d*)\" x2=\"(?P<x2>\d+\.\d*?)\" y2=\"(?P<y2>\d+\.?\d*)\".*')

	for i, line in enumerate(f):

		if i < 5: continue
		
		# For now skip paths, only check lines
		if not line.startswith("<line"): continue

		#path_match = re.match(path_pattern, line)
		line_match = re.match(line_pattern, line)

		x1 = float(line_match.group('x1'))
		y1 = float(line_match.group('y1'))
		x2 = float(line_match.group('x2'))
		y2 = float(line_match.group('y2'))

		p1 = (x1,y1)
		p2 = (x2,y2)

		edge = (p1,p2)

		coords = getLineCoords(edge)

		for point in coords:
			out_f.write(str(point[0]) + " " + str(point[1]) + ",")
			#filehandle.seek(-1, os.SEEK_END)
			#filehandle.truncate()
	
	in_f.close()
	out_f.close()
	
	return


def getLineCoords_bk(edge):
	x1 = edge[0][0]
	y1 = edge[0][1]
	x2 = edge[1][0]
	y2 = edge[1][1]
	
	coords = []

	m, b = getMB(x1,y1,x2,y2)

	if abs((x1-x2)) < abs((y1-y2)):
		for y in range(int(y1),int(y2)+1):
			x = int((y-b)/m)
			coords.append((x,y))
		return coords
	else:
		for x in range(int(x1),int(x2)+1):
			y = int(m*x+b)
			coords.append((x,y))
		return coords
