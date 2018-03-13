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


## 


	for ref_point in ref_reader:
		f.seek(0)
		comp_reader = f.readlines
		for comp_point in comp_reader:
		first_green = None
		second_green = None
		while y <= height:
			if np.array_equal(image[x,y], [0,255,0]) and first_green is None:
				first_green = y
			elif np.array_equal(image[x,y], [0,255,0]):
				second_green = y
				if abs(first_green - second_green) < MAX_DISTANCE:
					# put a box around this area
				next
			else:
				y += 1
				continue



#Old stuff from merging

				for box in enumerate(boxes):
					if centroid[1][1]+int(cluster_length/2) < box[1][1] and centroid[1][1]+int(cluster_length/2) > box[1][0]-SNAP_DIST:
						boxed = True
						boxes[box[0]][0] = centroid[1][1]-int(cluster_length/2)
						next
					elif centroid[1][1]-int(cluster_length/2) < box[1][1]+SNAP_DIST and centroid[1][1]-int(cluster_length/2) > box[1][0]:
						boxed = True
						boxes[box[0]][1] = centroid[1][1]+int(cluster_length/2)
						next
					else:
						continue
