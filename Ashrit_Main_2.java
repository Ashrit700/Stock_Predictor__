
@Entity
class Student {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private String course;

    
}


@Repository
interface StudentRepository extends JpaRepository<Student, Long> {
}


interface StudentService {
    Student saveStudent(Student s);
    List<Student> getStudents();
}

@Service
class StudentServiceImpl implements StudentService {

    @Autowired
    private StudentRepository repo;

    @Override
    public Student saveStudent(Student s) {
        return repo.save(s);
    }

    @Override
    public List<Student> getStudents() {
        return repo.findAll();
    }
}

@RestController
@RequestMapping("/students")
class StudentController {

    @Autowired
    private StudentService service;

    @PostMapping
    public Student addStudent(@RequestBody Student s) {
        return service.saveStudent(s);
    }

    @GetMapping
    public List<Student> getAll() {
        return service.getStudents();
    }
}
