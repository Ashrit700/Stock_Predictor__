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
    public Student save(Student student) {
        return repo.save(student);
    }
    @Override
    public List<Student> getAll() {
        return repo.findAll();
    }
    @Override
    public Student getById(Long id) {
        return repo.findById(id).orElse(null);
    }
    @Override
    public Student update(Long id, Student newData) {
        Student existing = repo.findById(id).orElse(null);
        if (existing != null) {
            existing.setName(newData.getName());
            existing.setCourse(newData.getCourse());
            return repo.save(existing);
        }
        return null;}
    @Override
    public String delete(Long id) {
        if (repo.existsById(id)) {
            repo.deleteById(id);
            return "Deleted";
        }
        return "Not Found";
    }
}
