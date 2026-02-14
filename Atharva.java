
// =======================
// 1. APPLICATION STARTER
// =======================
package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}


// =======================
// 2. MODEL / ENTITY LAYER
// =======================
package com.example.demo.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Student {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private String course;

    public Student() {}

    public Student(String name, String course) {
        this.name = name;
        this.course = course;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getCourse() { return course; }
    public void setCourse(String course) { this.course = course; }
}


// =====================================
// 3. PERSISTENCE LAYER (Repository)
// =====================================
package com.example.demo.repository;

import com.example.demo.model.Student;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface StudentRepository extends JpaRepository<Student, Long> {
}


// ======================
// 4. SERVICE LAYER (API)
// ======================
package com.example.demo.service;

import com.example.demo.model.Student;
import java.util.List;

public interface StudentService {
    Student saveStudent(Student student);
    List<Student> getAllStudents();
    Student getStudentById(Long id);
    Student updateStudent(Long id, Student student);
    String deleteStudent(Long id);
}


// =====================================
// 5. SERVICE IMPLEMENTATION LAYER
// =====================================
package com.example.demo.service;

import com.example.demo.model.Student;
import com.example.demo.repository.StudentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class StudentServiceImpl implements StudentService {

    @Autowired
    private StudentRepository repo;

    @Override
    public Student saveStudent(Student student) {
        return repo.save(student);
    }

    @Override
    public List<Student> getAllStudents() {
        return repo.findAll();
    }

    @Override
    public Student getStudentById(Long id) {
        return repo.findById(id).orElse(null);
    }

    @Override
    public Student updateStudent(Long id, Student newData) {
        Student s = repo.findById(id).orElse(null);
        if (s != null) {
            s.setName(newData.getName());
            s.setCourse(newData.getCourse());
            return repo.save(s);
        }
        return null;
    }

    @Override
    public String deleteStudent(Long id) {
        if (repo.existsById(id)) {
            repo.deleteById(id);
            return "Student deleted.";
        }
        return "Student not found.";
    }
}


// =========================
// 6. CONTROLLER LAYER (API)
// =========================
package com.example.demo.controller;

import com.example.demo.model.Student;
import com.example.demo.service.StudentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/students")
public class StudentController {

    @Autowired
    private StudentService service;

    // POST – Create
    @PostMapping
    public Student addStudent(@RequestBody Student student) {
        return service.saveStudent(student);
    }

    // GET – List all
    @GetMapping
    public List<Student> getAll() {
        return service.getAllStudents();
    }

    // GET – Get by ID
    @GetMapping("/{id}")
    public Student getOne(@PathVariable Long id) {
        return service.getStudentById(id);
    }

    // PUT – Update
    @PutMapping("/{id}")
    public Student update(@PathVariable Long id, @RequestBody Student s) {
        return service.updateStudent(id, s);
    }

    // DELETE – Delete
    @DeleteMapping("/{id}")
    public String delete(@PathVariable Long id) {
        return service.deleteStudent(id);
    }
}

